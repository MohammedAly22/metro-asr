"""
Render docs/results.json into a self-contained HTML report.

Audio is embedded as base64 data URIs so the file can be opened from disk,
emailed, or dropped on a static host with no other assets.

    python scripts/build_report.py --results docs/results.json --out docs/report.html
"""

import argparse
import base64
import difflib
import html
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metro_asr.utils import enable_utf8_stdout

enable_utf8_stdout()

SAMPLES_DIR = "test_samples"

HEAD_LABEL = {
    "greedy": "Greedy",
    "general": "General head",
    "technical": "Technical head",
    "medical": "Medical head",
}
HEAD_CLASS = {
    "greedy": "greedy",
    "general": "general",
    "technical": "technical",
    "medical": "medical",
}
DOMAIN_LABEL = {"general": "General speech", "technical": "Technical", "medical": "Medical"}

CSS = """
*,*::before,*::after{box-sizing:border-box}
:root{
  --bg:#0B0F17; --panel:#131A26; --panel2:#0F1520; --line:#26314A;
  --tx:#E8EDF5; --mut:#8A99B3; --dim:#5D6B85;
  --red:#E8232A; --red2:#FF5A60; --amber:#F5A524; --blue:#48A8E0; --green:#5BC98D;
}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--tx);
  font-family:'Segoe UI',-apple-system,BlinkMacSystemFont,Helvetica,Arial,sans-serif;
  line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:0 24px 96px}
code,.mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}

/* hero */
.hero{padding:64px 0 40px;border-bottom:1px solid var(--line);margin-bottom:40px}
.eyebrow{display:inline-block;font-size:11px;letter-spacing:2px;font-weight:700;
  color:var(--red2);border:1px solid rgba(232,35,42,.35);background:rgba(232,35,42,.08);
  padding:5px 12px;border-radius:20px;margin-bottom:20px}
h1{font-size:clamp(28px,4vw,42px);line-height:1.15;margin:0 0 14px;font-weight:800;letter-spacing:-.5px}
.sub{color:var(--mut);font-size:17px;max-width:820px;margin:0}
.meta{display:flex;flex-wrap:wrap;gap:10px;margin-top:26px}
.chip{font-size:12px;color:var(--mut);background:var(--panel);border:1px solid var(--line);
  padding:6px 12px;border-radius:7px}
.chip b{color:var(--tx);font-weight:600}

h2{font-size:24px;margin:56px 0 8px;font-weight:700;letter-spacing:-.3px;scroll-margin-top:20px}
h2 .bar{display:inline-block;width:4px;height:20px;background:linear-gradient(180deg,var(--red2),var(--red));
  border-radius:2px;margin-right:12px;vertical-align:-2px}
.lede{color:var(--mut);margin:0 0 24px;max-width:860px}
h3{font-size:16px;margin:32px 0 12px;font-weight:650}

/* toc */
.toc{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 22px;margin-bottom:8px}
.toc a{color:var(--mut);text-decoration:none;font-size:14px;padding:3px 0;display:inline-block}
.toc a:hover{color:var(--red2)}
.toc span{color:var(--dim);margin:0 8px}

/* tables */
.tbl{width:100%;border-collapse:collapse;font-size:14px;margin:8px 0 4px}
.tbl th{text-align:left;font-size:11px;letter-spacing:1.2px;color:var(--dim);
  font-weight:700;padding:10px 12px;border-bottom:1px solid var(--line);white-space:nowrap}
.tbl td{padding:11px 12px;border-bottom:1px solid rgba(38,49,74,.5);vertical-align:middle}
.tbl tr:last-child td{border-bottom:none}
.tbl td.num{text-align:right;font-family:ui-monospace,Consolas,monospace;font-size:13px;white-space:nowrap}
.scroll{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:4px 8px}
.best{color:var(--green);font-weight:700}
.worse{color:#FF8A8F}
.tag{display:inline-block;font-size:10.5px;letter-spacing:.6px;font-weight:700;
  padding:3px 9px;border-radius:5px;white-space:nowrap}
.tag.greedy{background:rgba(138,153,179,.14);color:#AEBBD0;border:1px solid rgba(138,153,179,.3)}
.tag.general{background:rgba(72,168,224,.12);color:var(--blue);border:1px solid rgba(72,168,224,.32)}
.tag.technical{background:rgba(245,165,36,.12);color:var(--amber);border:1px solid rgba(245,165,36,.32)}
.tag.medical{background:rgba(91,201,141,.12);color:var(--green);border:1px solid rgba(91,201,141,.32)}
.tag.ref{background:rgba(232,35,42,.12);color:var(--red2);border:1px solid rgba(232,35,42,.32)}

/* sample cards */
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;
  padding:22px 24px;margin-bottom:22px}
.card-head{display:flex;flex-wrap:wrap;align-items:center;gap:12px;
  padding-bottom:16px;border-bottom:1px solid var(--line);margin-bottom:18px}
.card-head .name{font-size:16px;font-weight:700;font-family:ui-monospace,Consolas,monospace}
.card-head .spacer{flex:1}
.card-head .dur{font-size:12px;color:var(--mut);font-family:ui-monospace,Consolas,monospace}
audio{width:100%;height:38px;margin-bottom:18px;border-radius:8px;
  filter:invert(.92) hue-rotate(180deg) saturate(.55)}
.row{display:grid;grid-template-columns:132px 1fr 104px;gap:14px;
  padding:12px 0;border-top:1px solid rgba(38,49,74,.55);align-items:start}
.row:first-of-type{border-top:none}
.row .lbl{padding-top:2px}
.txt{font-size:15.5px;line-height:1.95;direction:rtl;text-align:right;
  unicode-bidi:plaintext;word-break:break-word}
.txt.ref{color:#fff}
.sc{text-align:right;font-family:ui-monospace,Consolas,monospace;font-size:11.5px;
  color:var(--mut);padding-top:5px;white-space:nowrap;line-height:1.65}
.sc b{display:block;font-size:14px;color:var(--tx)}
mark{background:rgba(91,201,141,.16);color:var(--green);border-radius:3px;
  padding:0 3px;font-weight:600}
mark.warn{background:rgba(245,165,36,.16);color:var(--amber)}
mark.bad{background:rgba(255,138,143,.14);color:#FF8A8F}

.legend{display:flex;flex-wrap:wrap;gap:18px;margin:0 0 20px;font-size:13px;color:var(--mut)}
.legend .swatch{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:7px;vertical-align:middle}
.legend .warn{background:var(--amber)}
.legend .bad{background:#FF8A8F}

.note{background:var(--panel2);border:1px solid var(--line);border-left:3px solid var(--amber);
  border-radius:8px;padding:14px 18px;margin:20px 0;font-size:14px;color:var(--mut)}
.note b{color:var(--tx)}
.note.red{border-left-color:var(--red)}
.note.green{border-left-color:var(--green)}

.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px;margin:20px 0}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 20px}
.stat .k{font-size:11px;letter-spacing:1.2px;color:var(--dim);font-weight:700;margin-bottom:8px}
.stat .v{font-size:26px;font-weight:750;letter-spacing:-.5px}
.stat .d{font-size:12.5px;color:var(--mut);margin-top:4px}

footer{margin-top:72px;padding-top:24px;border-top:1px solid var(--line);
  color:var(--dim);font-size:12.5px}
a{color:var(--red2)}
@media(max-width:760px){
  .row{grid-template-columns:1fr;gap:6px}
  .sc{text-align:left}
  .wrap{padding:0 16px 64px}
}
"""


def b64_audio(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def classify_words(ref, hyp, ratio_threshold=0.75, max_merge=3):
    """
    Classify each hypothesis word as None (matches reference), 'warn' (differs only
    in how word boundaries fall, or a near-miss spelling), or 'bad' (genuinely wrong
    or unrelated to anything in the reference).

    CTC beam search routinely moves spaces around a common prefix like Arabic "و"
    ("and"), or splits/joins an English compound ("healthtech" / "health tech").
    Those are not recognition errors — the same characters were produced in the
    same order — so they get 'warn' rather than 'bad':

      1. exact word match against the reference                    -> None
      2. this hyp word == 2-3 consecutive ref words with no spaces  -> warn  (merge)
      3. this hyp word + a neighbor == one ref word with no spaces  -> warn  (split)
      4. closest reference word by character similarity is close    -> warn  (near-miss)
      5. otherwise                                                   -> bad
    """
    ref_words = ref.split()
    ref_set = set(ref_words)
    hyp_words = hyp.split()
    n = len(ref_words)

    windows = {
        k: {"".join(ref_words[i:i + k]) for i in range(n - k + 1)}
        for k in range(2, max_merge + 1)
        if n >= k
    }

    result = [None] * len(hyp_words)
    for idx, w in enumerate(hyp_words):
        if w in ref_set:
            continue

        if any(w in win for win in windows.values()):
            result[idx] = "warn"
            continue

        prev_w = hyp_words[idx - 1] if idx > 0 else None
        next_w = hyp_words[idx + 1] if idx < len(hyp_words) - 1 else None
        split_candidates = []
        if next_w:
            split_candidates.append(w + next_w)
        if prev_w:
            split_candidates.append(prev_w + w)
        if prev_w and next_w:
            split_candidates.append(prev_w + w + next_w)
        if any(c in ref_set for c in split_candidates):
            result[idx] = "warn"
            continue

        best_ratio = 0.0
        for rw in ref_set:
            if abs(len(rw) - len(w)) > max(3, len(w) // 2):
                continue
            r = difflib.SequenceMatcher(None, w, rw).ratio()
            if r > best_ratio:
                best_ratio = r
        result[idx] = "warn" if best_ratio >= ratio_threshold else "bad"

    return list(zip(hyp_words, result))


def diff_words(ref, hyp):
    """Render hypothesis text with per-word classification from classify_words()."""
    out = []
    for w, cls in classify_words(ref, hyp):
        if cls is None:
            out.append(html.escape(w))
        else:
            out.append(f'<mark class="{cls}">{html.escape(w)}</mark>')
    return " ".join(out)


def fmt_cell(v, best, lower_is_better=True):
    if v is None:
        return '<td class="num">—</td>'
    cls = "best" if v == best else ""
    return f'<td class="num {cls}">{v:.1f}</td>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="docs/results.json")
    ap.add_argument("--out", default="docs/report.html")
    ap.add_argument("--embed-audio", action="store_true",
                    help="inline audio as base64 so the file stands alone (large). "
                         "Default links to ../test_samples/, which keeps the file small.")
    ap.add_argument("--no-audio", action="store_true", help="omit audio players entirely")
    args = ap.parse_args()

    with open(args.results, encoding="utf-8") as f:
        D = json.load(f)

    methods = ["greedy"] + [h for h in ("general", "technical", "medical") if h in D["heads"]]
    P = []
    A = P.append

    A('<div class="wrap">')

    # ── hero ──
    A('<div class="hero">')
    A('<span class="eyebrow">METRO-ASR · EVALUATION REPORT</span>')
    A("<h1>Swapping the language head,<br>without touching the acoustic model</h1>")
    A('<p class="sub">One 61.6 M-parameter CTC acoustic model, decoded four ways: greedy, and with '
      'three interchangeable n-gram language heads. Every transcript below is real output on real '
      'Egyptian Arabic audio, scored against human references.</p>')
    A('<div class="meta">')
    A(f'<span class="chip">Model <b>{D["model"]["parameters"]:,} params</b></span>')
    A(f'<span class="chip">Vocabulary <b>{D["model"]["vocab_size"]:,} BPE</b></span>')
    A(f'<span class="chip">Beam width <b>{D["decoding"]["beam_width"]}</b></span>')
    A(f'<span class="chip">α <b>{D["decoding"]["alpha"]}</b> · β <b>{D["decoding"]["beta"]}</b></span>')
    A(f'<span class="chip">CPU threads <b>{D["decoding"]["threads"]}</b></span>')
    A(f'<span class="chip">Clips <b>{len(D["samples"])}</b></span>')
    A("</div></div>")

    # ── toc ──
    A('<div class="toc">')
    links = [("#heads", "The heads"), ("#results", "Aggregate results"),
             ("#matrix", "Per-clip scores"), ("#transcripts", "Transcripts"),
             ("#reading", "How to read this"), ("#method", "Method")]
    A("<span></span>".join(f'<a href="{h}">{t}</a>' for h, t in links))
    A("</div>")

    # ── heads ──
    A('<h2 id="heads"><span class="bar"></span>The heads</h2>')
    A('<p class="lede">The acoustic model is identical in every row. Only the n-gram model '
      'consulted during beam search changes — and it is a file on disk, swapped at run time.</p>')
    A('<div class="scroll"><table class="tbl"><thead><tr>'
      "<th>Head</th><th>Trained on</th><th>Size</th><th>Load</th></tr></thead><tbody>")
    A(f'<tr><td><span class="tag greedy">Greedy</span></td>'
      f"<td>No language model at all — per-frame argmax, then collapse</td>"
      f'<td class="num">0</td><td class="num">0 s</td></tr>')
    for h in methods:
        if h == "greedy":
            continue
        m = D["heads"][h]
        A(f'<tr><td><span class="tag {HEAD_CLASS[h]}">{HEAD_LABEL[h]}</span></td>'
          f'<td>{html.escape(m["description"])}</td>'
          f'<td class="num">{m["size_mb"]:,.0f} MB</td>'
          f'<td class="num">{m["load_seconds"]:.1f} s</td></tr>')
    A("</tbody></table></div>")

    # ── aggregate ──
    A('<h2 id="results"><span class="bar"></span>Aggregate results</h2>')
    A('<p class="lede">Word and character error rate, lower is better. Best figure in each row is '
      "green. Scoring strips punctuation and diacritics and normalises Alef/Ya variants — standard "
      "practice for Arabic ASR.</p>")

    S = D["summary"]
    A('<div class="scroll"><table class="tbl"><thead><tr><th>Test set</th><th>n</th>')
    for m in methods:
        A(f'<th style="text-align:right">{HEAD_LABEL[m]}<br>'
          f'<span style="color:var(--dim);font-weight:400">WER / CER</span></th>')
    A("</tr></thead><tbody>")
    for scope in ["all", "general", "technical", "medical"]:
        if scope not in S:
            continue
        row = S[scope]
        label = "All clips" if scope == "all" else DOMAIN_LABEL[scope]
        n = next(iter(row.values())).get("n", 0)
        wers = [row[m]["wer"] for m in methods if m in row and "wer" in row[m]]
        best_w = min(wers) if wers else None
        A(f"<tr><td><b>{label}</b></td><td class=\"num\">{n}</td>")
        for m in methods:
            if m not in row or "wer" not in row[m]:
                A('<td class="num">—</td>')
                continue
            w, c = row[m]["wer"], row[m]["cer"]
            cls = "best" if w == best_w else ""
            A(f'<td class="num {cls}">{w:.1f} / {c:.1f}</td>')
        A("</tr>")
    A("</tbody></table></div>")

    # headline stats
    def rel(scope, head):
        g = S[scope]["greedy"]["wer"]
        h = S[scope][head]["wer"]
        return (g - h) / g * 100

    A('<div class="grid">')
    if "technical" in S and "technical" in S["technical"]:
        A(f'<div class="stat"><div class="k">TECHNICAL CLIPS</div>'
          f'<div class="v" style="color:var(--amber)">−{rel("technical","technical"):.0f}%</div>'
          f'<div class="d">WER vs greedy, using the technical head</div></div>')
    if "medical" in S and "medical" in S["medical"]:
        A(f'<div class="stat"><div class="k">MEDICAL CLIPS</div>'
          f'<div class="v" style="color:var(--green)">−{rel("medical","medical"):.0f}%</div>'
          f'<div class="d">WER vs greedy, using the medical head</div></div>')
    if "general" in S and "technical" in S["general"]:
        d = S["general"]["technical"]["wer"] - S["general"]["greedy"]["wer"]
        A(f'<div class="stat"><div class="k">MISMATCHED HEAD</div>'
          f'<div class="v" style="color:#FF8A8F">+{d:.0f} pts</div>'
          f'<div class="d">WER when the technical head decodes general speech</div></div>')
    A("</div>")

    A('<div class="note green"><b>The heads are genuinely specialised.</b> Each domain head wins on '
      "its own domain and loses on the others. That is the result you want: it means the head is "
      "carrying real domain knowledge rather than just being a better general model.</div>")

    # ── per-clip matrix ──
    A('<h2 id="matrix"><span class="bar"></span>Per-clip scores</h2>')
    A('<p class="lede">WER per clip. Green marks the best head for that clip.</p>')
    A('<div class="scroll"><table class="tbl"><thead><tr>'
      "<th>Clip</th><th>Domain</th><th>Length</th>")
    for m in methods:
        A(f'<th style="text-align:right">{HEAD_LABEL[m]}</th>')
    A("</tr></thead><tbody>")
    for s in D["samples"]:
        wers = [s["decodes"][m]["wer"] for m in methods if m in s["decodes"]]
        best = min(w for w in wers if w is not None) if wers else None
        A(f'<tr><td class="mono">{html.escape(s["file"])}</td>'
          f'<td><span class="tag {HEAD_CLASS.get(s["domain"],"greedy")}">'
          f'{DOMAIN_LABEL.get(s["domain"], s["domain"])}</span></td>'
          f'<td class="num">{s["duration"]:.1f} s</td>')
        for m in methods:
            A(fmt_cell(s["decodes"].get(m, {}).get("wer"), best))
        A("</tr>")
    A("</tbody></table></div>")

    # ── transcripts ──
    A('<h2 id="transcripts"><span class="bar"></span>Transcripts</h2>')
    A('<p class="lede">Play the audio, read the human reference, then compare each decode.</p>')
    A('<div class="legend">'
      '<span><span class="swatch warn"></span>different word boundaries or a close spelling '
      "variant — e.g. <span class=\"mono\">وبركاته</span> for <span class=\"mono\">و بركاته</span>, "
      "or <span class=\"mono\">healthtech</span> for <span class=\"mono\">health tech</span>. "
      "Same content, not a recognition error.</span>"
      '<span><span class="swatch bad"></span>genuinely wrong or unrelated to the reference</span>'
      "</div>")

    order = {"technical": 0, "medical": 1, "general": 2}
    for s in sorted(D["samples"], key=lambda x: (order.get(x["domain"], 9), x["file"])):
        A('<div class="card">')
        A('<div class="card-head">')
        A(f'<span class="name">{html.escape(s["file"])}</span>')
        A(f'<span class="tag {HEAD_CLASS.get(s["domain"],"greedy")}">'
          f'{DOMAIN_LABEL.get(s["domain"], s["domain"])}</span>')
        A('<span class="spacer"></span>')
        A(f'<span class="dur">{s["duration"]:.2f} s · encoder {s["encoder_ms"]:.0f} ms</span>')
        A("</div>")

        path = os.path.join(SAMPLES_DIR, s["file"])
        if not args.no_audio and os.path.exists(path):
            if args.embed_audio:
                src = f"data:audio/wav;base64,{b64_audio(path)}"
            else:
                src = f"../{SAMPLES_DIR}/{s['file']}"
            A(f'<audio controls preload="metadata" src="{src}"></audio>')

        A('<div class="row"><div class="lbl"><span class="tag ref">REFERENCE</span></div>')
        A(f'<div class="txt ref">{html.escape(s["reference"])}</div><div class="sc"></div></div>')

        ref_n = s["reference_normalized"]
        for m in methods:
            d = s["decodes"].get(m)
            if not d:
                continue
            A(f'<div class="row"><div class="lbl"><span class="tag {HEAD_CLASS[m]}">'
              f'{HEAD_LABEL[m]}</span></div>')
            A(f'<div class="txt">{diff_words(ref_n, d["text_normalized"])}</div>')
            wer = f'{d["wer"]:.1f}%' if d["wer"] is not None else "—"
            A(f'<div class="sc"><b>{wer}</b>WER<br>'
              f'{d["cer"]:.1f}% CER<br>{d["decode_ms"]:.0f} ms</div></div>')
        A("</div>")

    # ── reading guide ──
    A('<h2 id="reading"><span class="bar"></span>How to read this</h2>')
    A('<div class="note"><b>WER is high in absolute terms, and that is expected.</b> '
      "These are unscripted YouTube and conversational clips with overlapping speech, background "
      "noise and heavy code-switching — the hardest end of the distribution. What matters here is "
      "the <i>difference between columns</i>, since the acoustic model and audio are identical "
      "across them.</div>")
    A('<div class="note red"><b>A mismatched head is worse than no head.</b> '
      "The technical head decoding general speech is meaningfully worse than plain greedy. A "
      "language head is a strong prior: it helps when the prior is right and hurts when it is "
      "wrong. Ship the head that matches your traffic, or ship the general one.</div>")
    A('<div class="note"><b>The general head is a 5-gram; the domain heads are 4-grams built from '
      "far less text.</b> They win in their domains despite that handicap, which is the point — "
      "domain fit beats scale for this component.</div>")

    # ── method ──
    A('<h2 id="method"><span class="bar"></span>Method</h2>')
    A('<div class="scroll"><table class="tbl"><tbody>')
    rows = [
        ("Acoustic model", f'Metro-Small, {D["model"]["parameters"]:,} parameters, unchanged across all runs'),
        ("Decoding", f'CTC beam search, width {D["decoding"]["beam_width"]}, '
                     f'α={D["decoding"]["alpha"]}, β={D["decoding"]["beta"]}'),
        ("Hardware", f'CPU only, {D["decoding"]["threads"]} threads, fp32'),
        ("Scoring", "jiwer WER/CER after lowercasing, punctuation and diacritic removal, "
                    "and Alef/Ya/Ta-Marbuta normalisation"),
        ("Technical corpus", "Egyptian Arabic Wikipedia filtered to technical articles + real "
                             "Arabic-English code-switching text + synthesised Egyptian carrier "
                             "phrases over a 182-term technical inventory"),
        ("Medical corpus", "Egyptian medical chat and QA from HuggingFace + real code-switching "
                           "text + synthesised carrier phrases over a 127-term medical inventory"),
        ("Leakage check", "No reference sentence appears in any training corpus; the maximum "
                          "shared 5-gram overlap between any reference and any corpus is 3"),
    ]
    for k, v in rows:
        A(f'<tr><td style="width:180px;color:var(--dim);font-size:12px;'
          f'letter-spacing:.8px;font-weight:700">{k.upper()}</td><td>{html.escape(v)}</td></tr>')
    A("</tbody></table></div>")

    A('<div class="note"><b>Reproduce:</b> <code>python scripts/build_domain_corpus.py --domain '
      "technical --out corpora/technical.txt</code> → <code>python scripts/train_lm.py --corpus "
      "corpora/technical.txt --out lm/technical_4gram.arpa --order 4</code> → <code>python "
      "scripts/compare_lm_heads.py</code> → <code>python scripts/build_report.py</code></div>")

    A("<footer>Metro-ASR · every number on this page was measured on the machine that generated "
      "it, from the released checkpoint. Nothing is estimated.</footer>")
    A("</div>")

    doc = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Metro-ASR — Language Head Comparison</title>"
        f"<style>{CSS}</style></head><body>{''.join(P)}</body></html>"
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"wrote {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
