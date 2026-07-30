"""
Pure-Python n-gram language model training, emitting standard ARPA files.

KenLM's `lmplz` is the right tool when you can run it, but it needs a C++
toolchain that is awkward on Windows and unavailable in many CI images. This
module builds an equivalent ARPA file with no compiler, no Boost and no Eigen.
The output is read by KenLM and by `pyctcdecode` exactly like an `lmplz` model.

Smoothing is **interpolated Witten-Bell**, which maps cleanly onto ARPA's
backoff representation. For a context h:

    lambda(h) = N1+(h) / (c(h) + N1+(h))

    P(w | h) = c(h,w) / (c(h) + N1+(h))  +  lambda(h) * P(w | h')

where c(h) is how often h occurs as a context and N1+(h) is the number of
distinct words that follow it. An unseen (h,w) falls back to exactly
lambda(h) * P(w | h'), so lambda(h) *is* the ARPA backoff weight for h — no
mass has to be renormalised after the fact.

Witten-Bell is chosen over modified Kneser-Ney deliberately: it is far more
robust on the small, uneven corpora that domain adaptation produces, where KN's
discount estimates become unstable.

Usage:
    from metro_asr.lm import build_arpa
    build_arpa(sentences, "lm/domain_5gram.arpa", order=5)
"""

import math
import re
from collections import defaultdict

BOS = "<s>"
EOS = "</s>"
UNK = "<unk>"

LOG0 = -99.0          # ARPA convention for "never generated"
_WS = re.compile(r"\s+")


def tokenize(line):
    """Whitespace tokenization on a lowercased line. Matches the ASR target space."""
    return _WS.split(line.strip().lower())


def _counts(sentences, order, min_count):
    """
    Count n-grams for every order up to `order`, over `<s> w1 ... wk </s>`.

    A single BOS is used, and every window at every order is counted. That
    guarantees ARPA's context-closure rule — the (n-1)-gram prefix of any
    counted n-gram is itself a counted window at the same position — which
    KenLM enforces strictly when loading.
    """
    ngrams = [defaultdict(int) for _ in range(order + 1)]   # index = n

    for sent in sentences:
        toks = sent if isinstance(sent, list) else tokenize(sent)
        toks = [t for t in toks if t]
        if not toks:
            continue
        padded = [BOS] + toks + [EOS]
        for n in range(1, order + 1):
            for i in range(len(padded) - n + 1):
                ngrams[n][tuple(padded[i:i + n])] += 1

    if min_count > 1:
        for n in range(2, order + 1):
            ngrams[n] = defaultdict(
                int, {g: c for g, c in ngrams[n].items() if c >= min_count}
            )
        _enforce_closure(ngrams, order)
    return ngrams


def _enforce_closure(ngrams, order):
    """
    Reinstate any lower-order gram that is the context — or the backed-off
    suffix — of a surviving higher-order gram. Pruning can otherwise leave
    an n-gram whose context was dropped, which makes the ARPA unloadable.
    """
    for n in range(order, 1, -1):
        for gram in list(ngrams[n].keys()):
            ctx, suffix = gram[:-1], gram[1:]
            if ctx not in ngrams[n - 1]:
                ngrams[n - 1][ctx] = 0
            if suffix not in ngrams[n - 1]:
                ngrams[n - 1][suffix] = 0


def _context_stats(ngrams, order):
    """c(h) and N1+(h) for every context h, keyed by tuple (() for unigram level)."""
    total = defaultdict(int)      # c(h) = sum_w c(h,w)
    distinct = defaultdict(int)   # N1+(h)

    for n in range(1, order + 1):
        for gram, count in ngrams[n].items():
            if count <= 0:
                continue
            # BOS is a context but is never itself predicted, so it must not
            # contribute mass to the unigram distribution.
            if n == 1 and gram[0] == BOS:
                continue
            h = gram[:-1]
            total[h] += count
            distinct[h] += 1
    return total, distinct


def build_arpa(sentences, output_path, order=5, min_count=1, progress=None):
    """
    Train an n-gram model and write it as ARPA.

    Args:
        sentences:   iterable of strings (or pre-tokenized lists)
        output_path: path to write; conventionally ends in .arpa
        order:       n-gram order
        min_count:   prune n-grams (order >= 2) seen fewer than this many times
        progress:    optional callable(str) for status messages

    Returns:
        dict of statistics.
    """
    say = progress or (lambda _m: None)

    say("counting n-grams")
    sentences = list(sentences)
    ngrams = _counts(sentences, order, min_count)
    total, distinct = _context_stats(ngrams, order)

    vocab = sorted({g[0] for g in ngrams[1]} | {UNK})
    V = len(vocab)
    say(f"vocabulary: {V:,} types")

    # ---- unigram level -------------------------------------------------
    # Backing off below unigrams means a uniform distribution over the vocabulary.
    uniform = 1.0 / V
    root_total = total[()]
    root_distinct = distinct[()]
    root_lambda = root_distinct / (root_total + root_distinct) if root_total else 1.0

    prob = {}          # gram -> probability (linear, not log)
    prob[(UNK,)] = root_lambda * uniform
    for gram, c in ngrams[1].items():
        if gram[0] == BOS:
            continue
        prob[gram] = c / (root_total + root_distinct) + root_lambda * uniform
    prob[(EOS,)] = prob.get((EOS,), root_lambda * uniform)

    # ---- higher orders, lowest first so the backoff term is available ----
    for n in range(2, order + 1):
        say(f"scoring {n}-grams ({len(ngrams[n]):,})")
        for gram, c in ngrams[n].items():
            h = gram[:-1]
            ht, hd = total[h], distinct[h]
            lam = hd / (ht + hd) if ht else 1.0

            lower = gram[1:]
            p_lower = prob.get(lower)
            if p_lower is None:
                # The backed-off gram was pruned; recurse down to the unigram.
                p_lower = prob.get((gram[-1],), prob[(UNK,)])
            prob[gram] = (c / (ht + hd) if ht else 0.0) + lam * p_lower

    # ---- backoff weights: lambda(h) for any gram used as a context ------
    def backoff(gram):
        ht, hd = total.get(gram, 0), distinct.get(gram, 0)
        if ht == 0:
            return 1.0          # never a context -> log10 = 0
        return hd / (ht + hd)

    # ---- write ----------------------------------------------------------
    say(f"writing {output_path}")
    by_order = {n: sorted(ngrams[n].keys()) for n in range(1, order + 1)}
    by_order[1] = sorted(set(by_order[1]) | {(UNK,)})

    def fmt(p):
        return f"{math.log10(p):.6f}" if p > 0 else f"{LOG0:.1f}"

    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\\data\\\n")
        for n in range(1, order + 1):
            f.write(f"ngram {n}={len(by_order[n])}\n")

        for n in range(1, order + 1):
            f.write(f"\n\\{n}-grams:\n")
            for gram in by_order[n]:
                if gram == (BOS,):
                    logp = f"{LOG0:.1f}"
                else:
                    logp = fmt(prob.get(gram, prob[(UNK,)]))
                text = " ".join(gram)
                if n < order:
                    bo = backoff(gram)
                    f.write(f"{logp}\t{text}\t{math.log10(bo):.6f}\n")
                else:
                    f.write(f"{logp}\t{text}\n")

        f.write("\n\\end\\\n")

    stats = {
        "order": order,
        "sentences": len(sentences),
        "vocab": V,
        "ngrams": {n: len(by_order[n]) for n in range(1, order + 1)},
        "path": output_path,
    }
    say("done")
    return stats


def perplexity(model_path, sentences):
    """
    Perplexity of `sentences` under an ARPA/binary model, via KenLM.

    Useful as an objective check that a domain head really is better matched to
    its domain than a general one.
    """
    import kenlm

    model = kenlm.Model(model_path)
    total_logprob = 0.0
    total_tokens = 0
    for sent in sentences:
        toks = tokenize(sent)
        if not toks:
            continue
        total_logprob += model.score(" ".join(toks), bos=True, eos=True)
        total_tokens += len(toks) + 1        # + </s>
    if total_tokens == 0:
        return float("inf")
    return 10 ** (-total_logprob / total_tokens)
