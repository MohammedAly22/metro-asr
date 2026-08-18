import os
import glob
import hashlib

import gradio as gr

from metro_asr import MetroASREngine

# ========================= CONFIGURATION =========================
DEVICE = "cpu"

HERE = os.path.dirname(os.path.abspath(__file__))

# Gradio moved `theme` and `css` from Blocks() to launch() in v6. This app is
# shipped both to the GitHub repo (dev machines, currently 6.x) and to the
# HuggingFace Space (pinned to 5.x), so pick the right home for them at import
# time rather than committing two divergent copies of the file.
GRADIO_MAJOR = int(gr.__version__.split(".")[0])


def _resolve_model_dir():
    """
    Locate the checkpoint. `METRO_MODEL` wins; otherwise look for the two
    layouts this app actually ships in — `checkpoints/` in the git repo and
    `model_files/` on the Space — before falling back to the HF alias.
    """
    env = os.environ.get("METRO_MODEL")
    if env:
        return env
    for name in ("checkpoints", "model_files"):
        candidate = os.path.join(HERE, name)
        if os.path.exists(os.path.join(candidate, "config.yaml")):
            return candidate
    return "small"


MODEL = _resolve_model_dir()

# The 5.9 GB KenLM binary is not checked into either repo. `auto` picks it up if
# it happens to sit next to the checkpoint; otherwise fetch it once on cold
# start. METRO_SKIP_LM=1 forces a greedy-only boot, which is much faster while
# iterating on the UI.
LM_REPO_ID = os.environ.get("METRO_LM_REPO", "mohammedaly22/Metro-ASR-Small")
LM_FILENAME = "lm_5gram.bin"
SKIP_LM = os.environ.get("METRO_SKIP_LM") == "1"

SERVER_PORT = int(os.environ.get("METRO_PORT", 7860))

# Colab, Kaggle and other hosted notebooks have no direct route to localhost, so
# 0.0.0.0:PORT is unreachable there — set METRO_SHARE=true to get a public
# gradio.live tunnel URL printed to stdout instead. Leave unset for local use.
SHARE = os.environ.get("METRO_SHARE", "false").lower() == "true"

EXAMPLES_DIR = os.path.join(HERE, "gradio_examples")
# =================================================================


def _resolve_lm_path():
    if SKIP_LM:
        return None

    if os.path.isdir(MODEL):
        local = os.path.join(MODEL, LM_FILENAME)
        if os.path.exists(local):
            return local

    try:
        from huggingface_hub import hf_hub_download
        print(f"Fetching {LM_FILENAME} from {LM_REPO_ID} (~5.9 GB, once per cold start)...")
        return hf_hub_download(repo_id=LM_REPO_ID, filename=LM_FILENAME)
    except Exception as exc:
        print(f"Could not fetch {LM_FILENAME}: {exc} — falling back to greedy decoding.")
        return None


print(f"Loading Metro-ASR from {MODEL}...")
engine = MetroASREngine.from_pretrained(MODEL, device=DEVICE, lm_path=_resolve_lm_path())
_param_count = engine.param_count
print(f"Model loaded: {_param_count:,} params | LM: {'loaded' if engine.has_lm else 'none'}")


# ── Examples ──
# Labelled by language and duration so the picker is scannable. Files are
# discovered at startup, so a missing or renamed clip drops out of the list
# instead of breaking the app.

EXAMPLE_LANGUAGES = [
    ("ar_", "Egyptian Arabic"),
    ("cs_", "Code-Switching"),
    ("en_", "English"),
]


def _example_duration(path):
    try:
        import soundfile as sf
        return sf.info(path).duration
    except Exception:
        return None


def _build_examples():
    found = sorted(glob.glob(os.path.join(EXAMPLES_DIR, "*")))
    if not found:
        return [], []

    by_prefix = {prefix: [] for prefix, _ in EXAMPLE_LANGUAGES}
    for path in found:
        name = os.path.basename(path).lower()
        for prefix, _ in EXAMPLE_LANGUAGES:
            if name.startswith(prefix):
                by_prefix[prefix].append(path)
                break

    # The same clip filed under two language prefixes would appear twice under
    # contradictory labels, so keep the first in EXAMPLE_LANGUAGES order and
    # drop later copies. Compares content, not filename.
    seen = set()
    paths, labels = [], []
    for prefix, language in EXAMPLE_LANGUAGES:
        index = 0
        for path in by_prefix[prefix]:
            try:
                with open(path, "rb") as fh:
                    digest = hashlib.md5(fh.read()).hexdigest()
            except OSError:
                continue
            if digest in seen:
                print(f"Skipping duplicate example: {os.path.basename(path)}")
                continue
            seen.add(digest)

            index += 1
            duration = _example_duration(path)
            suffix = f" · {duration:.0f}s" if duration else ""
            paths.append([path])
            labels.append(f"{language} {index}{suffix}")
    return paths, labels


EXAMPLE_PATHS, EXAMPLE_LABELS = _build_examples()


# ── File upload transcription ──

def transcribe(audio_path, decoding_method, beam_width, lm_alpha, lm_beta):
    if audio_path is None:
        return "", "", ""

    try:
        use_beam = decoding_method == "Beam Search + LM"

        result = engine.transcribe(
            audio_path,
            beam_search=use_beam,
            beam_width=int(beam_width),
            lm_alpha=lm_alpha,
            lm_beta=lm_beta,
        )

        text = result.text
        duration = result.duration
        model_time = result.inference_time
        decode_time = result.decoding_time
        total_time = model_time + decode_time
        rtf = result.rtf

        if use_beam and not engine.has_lm:
            method_label = "Greedy (LM unavailable)"
        elif use_beam:
            method_label = "Beam Search + LM"
        else:
            method_label = "Greedy"

        if 0 < rtf < 1.0:
            speed_note = (
                f'<div class="speed-badge"><span class="speed-val">{1/rtf:.1f}x</span>'
                ' faster than real-time</div>'
            )
        else:
            speed_note = (
                f'<div class="speed-badge"><span class="speed-val">{rtf:.1f}x</span>'
                ' slower than real-time</div>'
            )

        # Beam parameters get their own rows rather than being appended to the
        # method string — a long single value is what used to force the metrics
        # panel wider than its column.
        rows = [
            ("Audio Duration", f"{duration:.2f}s"),
            ("Model Inference", f"{model_time*1000:.1f}ms"),
            ("Decoding Time", f"{decode_time*1000:.1f}ms"),
            ("Total Latency", f"{total_time*1000:.1f}ms"),
            ("RTF", f"{rtf:.4f}"),
            ("Method", method_label),
        ]
        if use_beam and engine.has_lm:
            rows += [
                ("Beam Width", f"{int(beam_width)}"),
                ("Alpha / Beta", f"{lm_alpha} / {lm_beta}"),
            ]
        rows += [
            ("Device", DEVICE.upper()),
            ("Parameters", f"{_param_count:,}"),
        ]

        metrics_rows = "".join(f"""
            <div class="metric-row">
                <span class="metric-label">{label}</span>
                <span class="metric-value">{value}</span>
            </div>""" for label, value in rows)
        stats_html = f'<div class="metrics-container">{metrics_rows}</div>'

        return text, speed_note, stats_html

    except Exception as e:
        return f"Error: {str(e)}", "", ""


def on_decoding_change(method):
    return gr.update(visible=(method == "Beam Search + LM"))


# ═══════════════════════ THEME & CSS ═══════════════════════

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Global ── */
body, .gradio-container {
    background: #0a0a0a !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ══ Width lock ══
   Three separate things used to widen the page: the audio waveform for a long
   clip, the example table, and long metric values. They share one root cause —
   flex and grid children default to `min-width: auto`, so they refuse to shrink
   below their content and push every ancestor wider. Pinning the container and
   resetting min-width down the tree is what actually holds the layout still. */
html, body {
    max-width: 100% !important;
    overflow-x: hidden !important;
}
.gradio-container {
    max-width: 940px !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding-left: 16px !important;
    padding-right: 16px !important;
    overflow-x: hidden !important;
}
.gradio-container .block,
.gradio-container .form,
.gradio-container .wrap,
.gradio-container .panel,
.gradio-container .column,
.gradio-container .row,
.gradio-container > .main,
.gradio-container > .main > .wrap,
.gradio-container .gradio-row,
.gradio-container .gradio-column {
    min-width: 0 !important;
    max-width: 100% !important;
}
/* Media never dictates layout width. */
.gradio-container canvas,
.gradio-container audio,
.gradio-container img,
.gradio-container video,
.gradio-container svg {
    max-width: 100% !important;
}
/* Anything legitimately wide scrolls inside its own box. */
.gradio-container table,
.gradio-container .table-wrap {
    max-width: 100% !important;
    overflow-x: auto !important;
}

/* ── Title ── */
.metro-title {
    text-align: center;
    font-size: 2.8em;
    font-weight: 900;
    letter-spacing: 6px;
    color: #fff;
    font-family: 'Space Grotesk', sans-serif;
    margin: 20px 0 8px 0;
    background: linear-gradient(135deg, #ff1744 0%, #e53935 50%, #b71c1c 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ── Tags ── */
.metro-tags {
    text-align: center;
    margin-bottom: 28px;
}
.metro-tags span {
    display: inline-block;
    background: rgba(229, 57, 53, 0.08);
    border: 1px solid rgba(229, 57, 53, 0.25);
    color: #e57373;
    padding: 5px 16px;
    border-radius: 20px;
    font-size: 0.7em;
    letter-spacing: 1.8px;
    margin: 3px 4px;
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
}

/* ── Two-Column Layout ── */
.main-row {
    gap: 28px !important;
    align-items: flex-start !important;
}
@media (max-width: 860px) {
    .main-row {
        flex-direction: column !important;
    }
}

/* ── Output Text ── */
/* `unicode-bidi: plaintext` picks direction per line from its first strong
   character, so Arabic renders right-to-left and English left-to-right in the
   same box — the demo transcribes both. */
.output-text textarea {
    font-size: 1.15em !important;
    line-height: 1.9 !important;
    direction: ltr !important;
    unicode-bidi: plaintext !important;
    text-align: start !important;
    min-height: 220px !important;
    background: #0d0d0d !important;
    color: #f0f0f0 !important;
    border: 1px solid #252525 !important;
    border-radius: 12px !important;
    padding: 20px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    overflow-wrap: anywhere !important;
}

/* ── Speed Badge ── */
.speed-badge {
    text-align: center;
    padding: 14px 0 8px 0;
    font-family: 'Space Grotesk', sans-serif;
    color: #ccc;
    font-size: 0.95em;
}
.speed-badge .speed-val {
    color: #ff5252;
    font-size: 1.25em;
    font-weight: 700;
}

/* ── Metrics ── */
.metrics-container {
    background: #0e0e0e;
    border: 1px solid #1a1a1a;
    border-radius: 12px;
    padding: 16px 20px;
    margin-top: 4px;
}
/* Grid tracks with an explicit 0 minimum, so a long value wraps instead of
   stretching the panel. */
.metric-row {
    display: grid;
    grid-template-columns: minmax(0, auto) minmax(0, 1fr);
    gap: 4px 14px;
    align-items: baseline;
    padding: 8px 0;
    border-bottom: 1px solid #151515;
    font-size: 0.88em;
    font-family: 'Space Grotesk', sans-serif;
}
.metric-row:last-child {
    border-bottom: none;
}
.metric-label {
    color: #e57373;
    font-weight: 600;
}
.metric-value {
    color: #aaa;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.92em;
    text-align: right;
    overflow-wrap: anywhere;
    min-width: 0;
}

/* ── Button ── */
.transcribe-btn button {
    background: linear-gradient(135deg, #b71c1c 0%, #e53935 50%, #ff1744 100%) !important;
    box-shadow: 0 4px 24px rgba(229, 57, 53, 0.25) !important;
    letter-spacing: 3px !important;
    font-weight: 700 !important;
    font-size: 1em !important;
    font-family: 'Space Grotesk', sans-serif !important;
    border: none !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}
.transcribe-btn button:hover {
    box-shadow: 0 4px 40px rgba(229, 57, 53, 0.5) !important;
    transform: translateY(-1px);
}

/* ── Beam Section ── */
.beam-section {
    border: 1px solid #1e1e1e !important;
    border-radius: 12px !important;
    padding: 14px !important;
    background: #0e0e0e !important;
    margin-top: 6px !important;
}

/* ── Examples ── */
.examples-panel {
    margin-top: 4px !important;
}
.examples-panel .gallery-item,
.examples-panel button.gallery-item {
    background: #111 !important;
    border: 1px solid #242424 !important;
    color: #ccc !important;
    border-radius: 8px !important;
    font-size: 0.8em !important;
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: 0.4px !important;
    white-space: nowrap !important;
}
.examples-panel .gallery-item:hover,
.examples-panel button.gallery-item:hover {
    border-color: #e53935 !important;
    color: #fff !important;
}
/* The example list is the one place a horizontal scrollbar is correct — it
   keeps a long row of chips from widening the page. */
.examples-panel .gallery,
.examples-panel .table-wrap {
    max-width: 100% !important;
    overflow-x: auto !important;
}

/* ── About / Footer ── */
.about-section {
    background: #0e0e0e !important;
    border: 1px solid #1a1a1a !important;
    border-radius: 12px !important;
    padding: 18px !important;
    color: #888 !important;
}
.about-section * {
    overflow-wrap: anywhere !important;
}
.metro-footer {
    text-align: center;
    color: #2a2a2a;
    font-size: 0.72em;
    padding: 28px 0 10px 0;
    letter-spacing: 2.5px;
    font-family: 'Space Grotesk', sans-serif;
}

footer { visibility: hidden !important; }
"""

METRO_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.red,
    secondary_hue=gr.themes.colors.red,
    neutral_hue=gr.themes.colors.gray,
    font=gr.themes.GoogleFont("Space Grotesk"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    body_background_fill="#0a0a0a",
    body_background_fill_dark="#0a0a0a",
    body_text_color="#e0e0e0",
    body_text_color_dark="#e0e0e0",
    body_text_color_subdued="#888",
    block_background_fill="#111111",
    block_background_fill_dark="#111111",
    block_border_color="#1e1e1e",
    block_border_color_dark="#1e1e1e",
    block_label_text_color="#ccc",
    block_label_text_color_dark="#ccc",
    block_title_text_color="#e0e0e0",
    block_title_text_color_dark="#e0e0e0",
    input_background_fill="#0d0d0d",
    input_background_fill_dark="#0d0d0d",
    input_border_color="#333",
    input_border_color_dark="#333",
    input_border_color_focus="#ff1a1a",
    input_border_color_focus_dark="#ff1a1a",
    button_primary_background_fill="linear-gradient(135deg, #b71c1c 0%, #e53935 50%, #ff1744 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #d32f2f 0%, #ff1744 100%)",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#1a1a1a",
    button_secondary_background_fill_hover="#252525",
    button_secondary_text_color="#ccc",
    slider_color="#e53935",
    checkbox_background_color="#1a1a1a",
    checkbox_background_color_selected="#e53935",
    checkbox_label_text_color="#ccc",
    border_color_primary="#333",
    block_radius="12px",
)

_UI_KWARGS = {"theme": METRO_THEME, "css": CSS}
_BLOCKS_KWARGS = {} if GRADIO_MAJOR >= 6 else dict(_UI_KWARGS)
_LAUNCH_KWARGS = dict(_UI_KWARGS) if GRADIO_MAJOR >= 6 else {}


# ═══════════════════════ UI LAYOUT ═══════════════════════

with gr.Blocks(title="Metro-ASR", fill_width=False, **_BLOCKS_KWARGS) as demo:

    # ── Title ──
    gr.HTML('<div class="metro-title">Metro-ASR</div>')

    # ── Tags ──
    gr.HTML(
        '<div class="metro-tags">'
        '<span>CONFORMER</span> <span>CTC</span> <span>KENLM</span> '
        f'<span>CPU OPTIMIZED</span> <span>{_param_count:,} PARAMS</span>'
        '</div>'
    )

    with gr.Row(elem_classes=["main-row"]):

        # ─────────── Left: input ───────────
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                type="filepath",
                label="Upload or record audio",
                sources=["upload", "microphone"],
                waveform_options=gr.WaveformOptions(
                    waveform_color="#e53935",
                    waveform_progress_color="#ff5252",
                ),
            )

            if EXAMPLE_PATHS:
                gr.Examples(
                    examples=EXAMPLE_PATHS,
                    inputs=[audio_input],
                    example_labels=EXAMPLE_LABELS,
                    label="Examples",
                    examples_per_page=12,
                    # Caching would run every clip through beam search at build
                    # time, which on a cpu-basic Space is slow enough to time out.
                    cache_examples=False,
                    elem_id="metro-examples",
                )

            decoding_method = gr.Radio(
                choices=["Greedy", "Beam Search + LM"],
                value="Beam Search + LM" if engine.has_lm else "Greedy",
                label="Decoding",
                info="Greedy is instant; Beam + LM is more accurate",
            )

            with gr.Group(visible=True, elem_classes=["beam-section"]) as beam_group:
                beam_width = gr.Slider(
                    minimum=5, maximum=500, value=100, step=5,
                    label="Beam Width",
                    info="Parallel hypotheses",
                )
                lm_alpha = gr.Slider(
                    minimum=0.0, maximum=3.0, value=0.5, step=0.1,
                    label="LM Weight (Alpha)",
                    info="Language model influence",
                )
                lm_beta = gr.Slider(
                    minimum=0.0, maximum=10.0, value=5.0, step=0.5,
                    label="Word Bonus (Beta)",
                    info="Prevents word deletion",
                )

            submit_btn = gr.Button(
                "TRANSCRIBE",
                variant="primary",
                size="lg",
                elem_classes=["transcribe-btn"],
            )

        # ─────────── Right: output ───────────
        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="Transcription",
                lines=9,
                elem_classes=["output-text"],
                placeholder="Pick an example or upload audio, then press TRANSCRIBE.",
            )

            speed_badge = gr.HTML("")
            stats_output = gr.HTML("")

    # ── About ──
    with gr.Accordion("About Metro-ASR", open=False):
        gr.Markdown(
            "**Metro-ASR** is a non-autoregressive ASR system built on a Conformer "
            "acoustic encoder with RoPE, SwiGLU, RMSNorm, SE-Conv, Stochastic Depth "
            "and intermediate CTC supervision, paired with a detachable KenLM "
            "language head. Trained on 130K+ Egyptian Arabic audio samples with "
            "code-switching support. This demo runs entirely on **CPU**.",
            elem_classes=["about-section"],
        )

    gr.HTML(
        '<div class="metro-footer">'
        'METRO-ASR &mdash; CONFORMER + CTC + KENLM &bull; CPU-OPTIMIZED'
        '</div>'
    )

    # ── Events ──
    decoding_method.change(
        fn=on_decoding_change,
        inputs=[decoding_method],
        outputs=[beam_group],
    )

    submit_btn.click(
        fn=transcribe,
        inputs=[audio_input, decoding_method, beam_width, lm_alpha, lm_beta],
        outputs=[output_text, speed_badge, stats_output],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=SERVER_PORT,
        share=SHARE,
        ssr_mode=False,
        **_LAUNCH_KWARGS,
    )
