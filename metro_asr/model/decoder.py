"""
CTC Beam Search Decoder with optional KenLM language model.

Greedy decoding is fast but makes independent per-frame decisions.
Beam search + LM rescoring produces valid words by leveraging language context.
"""
import torch
import numpy as np


def load_unigrams_from_arpa(arpa_path, max_unigrams=2_000_000):
    """
    Read the \\1-grams section of an ARPA file as UTF-8.

    pyctcdecode does this itself, but opens the file with the platform's
    default encoding — which mangles Arabic on any system where that is not
    UTF-8 (every Windows install). Reading it here keeps decoding correct and
    lets the decoder score word boundaries properly.
    """
    unigrams = set()
    in_section = False
    with open(arpa_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("\\1-grams:"):
                in_section = True
                continue
            if line.startswith("\\") and in_section:
                break
            if not in_section or not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                word = parts[1].strip()
                if word and not (word.startswith("<") and word.endswith(">")):
                    unigrams.add(word)
            if len(unigrams) >= max_unigrams:
                break
    return unigrams


class CTCBeamSearchDecoder:
    def __init__(self, tokenizer, lm_path=None, beam_width=20, alpha=0.5, beta=1.0):
        from pyctcdecode import build_ctcdecoder

        self.tokenizer = tokenizer
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta

        vocab = self._build_vocab()

        if lm_path:
            unigrams = None
            if str(lm_path).lower().endswith(".arpa"):
                unigrams = load_unigrams_from_arpa(lm_path)
            self.decoder = build_ctcdecoder(
                labels=vocab,
                kenlm_model_path=lm_path,
                unigrams=unigrams,
                alpha=alpha,
                beta=beta,
            )
            self.has_lm = True
        else:
            self.decoder = build_ctcdecoder(labels=vocab)
            self.has_lm = False

    def _build_vocab(self):
        """
        Model-id-ordered labels for pyctcdecode.

        SentencePiece pieces keep their leading "▁" so pyctcdecode detects a
        subword vocabulary and merges pieces into words before scoring them
        against the n-gram LM. Duplicate labels would be ambiguous, so
        collisions are given a unique placeholder.
        """
        vocab = []
        seen = set()
        for i in range(self.tokenizer.vocab_size):
            if i == self.tokenizer.blank_id:
                token = ""
            elif i == self.tokenizer.unk_id:
                token = "⁇"
            elif i == self.tokenizer.pad_id:
                token = "⁈"
            else:
                token = self.tokenizer.id_to_piece(i)

            if token in seen and token != "":
                token = f"<dup_{i}>"
            seen.add(token)
            vocab.append(token)
        return vocab

    def _apply_params(self, alpha=None, beta=None):
        """
        Retune the LM weights in place.

        pyctcdecode bakes alpha and beta into the decoder when it is built, but
        exposes ``reset_params`` to change them afterwards. Going through it is
        what keeps per-call tuning free: rebuilding the decoder would reload the
        KenLM binary — several GB for the shipped 5-gram — on every change.
        It is a no-op when no language model is attached.
        """
        a = self.alpha if alpha is None else alpha
        b = self.beta if beta is None else beta
        if (a, b) == (self.alpha, self.beta):
            return
        self.decoder.reset_params(alpha=a, beta=b)
        self.alpha, self.beta = a, b

    def decode(self, log_probs, lengths=None, beam_width=None, alpha=None, beta=None):
        self._apply_params(alpha, beta)
        width = beam_width or self.beam_width

        results = []
        log_probs_np = log_probs.float().cpu().numpy()

        for i in range(log_probs_np.shape[0]):
            max_t = lengths[i].item() if lengths is not None else log_probs_np.shape[1]
            lp = log_probs_np[i, :max_t, :]

            text = self.decoder.decode(
                lp,
                beam_width=width,
            )
            results.append(text.strip())

        return results

    def decode_batch(self, log_probs, lengths=None, batch_size=16,
                     beam_width=None, alpha=None, beta=None):
        self._apply_params(alpha, beta)
        width = beam_width or self.beam_width

        log_probs_np = log_probs.float().cpu().numpy()

        if lengths is not None:
            pool_input = []
            for i in range(log_probs_np.shape[0]):
                max_t = lengths[i].item()
                pool_input.append(log_probs_np[i, :max_t, :])
        else:
            pool_input = [log_probs_np[i] for i in range(log_probs_np.shape[0])]

        with self.decoder.pool(processes=min(4, len(pool_input))) as pool:
            results = pool.decode_batch(
                pool_input,
                beam_width=width,
            )

        return [r.strip() for r in results]


def build_decoder(tokenizer, lm_path=None, beam_width=20, alpha=0.5, beta=1.0):
    if lm_path:
        return CTCBeamSearchDecoder(
            tokenizer, lm_path=lm_path,
            beam_width=beam_width, alpha=alpha, beta=beta,
        )
    return None
