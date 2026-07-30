import os
import numpy as np
import torch
import gradio as gr

from metro_asr import MetroASREngine

# ========================= CONFIGURATION =========================
DEVICE = "cpu"

# A local directory (config.yaml + model.pt + bpe.model), a size alias
# ("small"), or a HuggingFace repo id.
MODEL = os.environ.get("METRO_MODEL", "checkpoints")

# "auto" picks up a KenLM binary next to the checkpoint; None disables beam search.
LM_PATH = os.environ.get("METRO_LM", "auto")

SERVER_PORT = int(os.environ.get("METRO_PORT", 7860))
STREAM_MIN_DURATION = 3.0
STREAM_TARGET_SR = 16000
# =================================================================

print(f"Loading Metro-ASR from {MODEL}...")
engine = MetroASREngine.from_pretrained(MODEL, device=DEVICE, lm_path=LM_PATH)
_param_count = engine.param_count
print(f"Model loaded: {_param_count:,} params | LM: {'loaded' if engine.has_lm else 'none'}")


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
            method_label = "Greedy (LM not available)"
        elif use_beam:
            method_label = f"Beam Search + LM (beam={int(beam_width)}, alpha={lm_alpha}, beta={lm_beta})"
        else:
            method_label = "Greedy Decoding"

        if rtf < 1.0 and rtf > 0:
            speed_note = f'<div class="speed-badge"><span class="speed-val">{1/rtf:.1f}x</span> faster than real-time</div>'
        else:
            speed_note = f'<div class="speed-badge"><span class="speed-val">{rtf:.1f}x</span> slower than real-time</div>'

        metrics_rows = "".join(f"""
            <div class="metric-row">
                <span class="metric-label">{label}</span>
                <span class="metric-value">{value}</span>
            </div>""" for label, value in [
            ("Audio Duration", f"{duration:.2f}s"),
            ("Model Inference", f"{model_time*1000:.1f}ms"),
            ("Decoding Time", f"{decode_time*1000:.1f}ms"),
            ("Total Latency", f"{total_time*1000:.1f}ms"),
            ("RTF", f"{rtf:.4f}"),
            ("Method", method_label),
            ("Device", DEVICE.upper()),
            ("Parameters", f"{_param_count:,}"),
        ])
        stats_html = f'<div class="metrics-container">{metrics_rows}</div>'

        return text, speed_note, stats_html

    except Exception as e:
        return f"Error: {str(e)}", "", ""


def on_decoding_change(method):
    return gr.update(visible=(method == "Beam Search + LM"))


# ── Streaming transcription ──
# Gradio streaming sends tiny chunks (~200ms). CTC needs at least ~3s of
# context. We accumulate all audio received so far and re-transcribe the
# full buffer each time — the model is fast enough on CPU for this.

def _raw_to_float(waveform):
    if waveform.dtype == np.int16:
        return waveform.astype(np.float32) / 32768.0
    if waveform.dtype == np.int32:
        return waveform.astype(np.float32) / 2147483648.0
    if waveform.dtype == np.float64:
        return waveform.astype(np.float32)
    return waveform.astype(np.float32)


def _transcribe_waveform(waveform_np):
    waveform = torch.tensor(waveform_np, dtype=torch.float32)
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0)
    if waveform.shape[0] == 0:
        return ""

    return engine.transcribe(waveform).text.strip()


def transcribe_streaming(audio_chunk, state):
    if state is None:
        state = {"buffer": np.array([], dtype=np.float32), "sr": STREAM_TARGET_SR}

    if audio_chunk is None:
        text = _transcribe_waveform(state["buffer"]) if state["buffer"].size > 0 else ""
        return text, state

    sr, chunk = audio_chunk
    chunk = _raw_to_float(chunk)

    if chunk.ndim > 1:
        chunk = chunk.mean(axis=1)

    if sr != STREAM_TARGET_SR:
        import torchaudio
        chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=STREAM_TARGET_SR)
        chunk = resampler(chunk_tensor).squeeze(0).numpy()
        state["sr"] = STREAM_TARGET_SR

    state["buffer"] = np.concatenate([state["buffer"], chunk])

    buffer_duration = state["buffer"].shape[0] / STREAM_TARGET_SR
    if buffer_duration < STREAM_MIN_DURATION:
        return "", state

    text = _transcribe_waveform(state["buffer"])
    return text, state


def clear_stream():
    return "", None


# ═══════════════════════ THEME & CSS ═══════════════════════

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Global ── */
body, .gradio-container {
    background: #0a0a0a !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
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

/* ── Section Labels ── */
.section-label {
    color: #e57373 !important;
    font-size: 0.78em !important;
    letter-spacing: 2.5px !important;
    font-weight: 700 !important;
    padding-bottom: 10px !important;
    border-bottom: 1px solid #1a1a1a !important;
    margin-bottom: 12px !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── Two-Column Layout ── */
.main-row {
    gap: 32px !important;
}
.main-row > div {
    flex: 1 1 50% !important;
    min-width: 0 !important;
}

/* ── Output Text ── */
.output-text textarea {
    font-size: 1.2em !important;
    line-height: 2 !important;
    direction: rtl !important;
    text-align: right !important;
    min-height: 200px !important;
    background: #0d0d0d !important;
    color: #f0f0f0 !important;
    border: 1px solid #252525 !important;
    border-radius: 12px !important;
    padding: 20px !important;
    font-family: 'Space Grotesk', sans-serif !important;
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
.metric-row {
    display: flex;
    justify-content: space-between;
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

/* ── Streaming ── */
.stream-output textarea {
    font-size: 1.3em !important;
    line-height: 1.9 !important;
    direction: rtl !important;
    text-align: right !important;
    min-height: 220px !important;
    background: #0d0d0d !important;
    color: #f0f0f0 !important;
    border: 1px solid #252525 !important;
    border-radius: 12px !important;
    padding: 24px !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
.pulse-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    background: #e53935;
    border-radius: 50%;
    margin-right: 8px;
    animation: pulse 1.5s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.8); }
}
.stream-hint {
    text-align: center;
    color: #666;
    font-size: 0.82em;
    margin-bottom: 12px;
}
.clear-btn button {
    background: #1a1a1a !important;
    border: 1px solid #333 !important;
    color: #aaa !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
}

/* ── About / Footer ── */
.about-section {
    background: #0e0e0e !important;
    border: 1px solid #1a1a1a !important;
    border-radius: 12px !important;
    padding: 18px !important;
    color: #888 !important;
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


# ═══════════════════════ UI LAYOUT ═══════════════════════

with gr.Blocks(title="Metro-ASR") as demo:

    # ── Title ──
    gr.HTML(
        '<div class="metro-title">Metro-ASR</div>'
    )

    # ── Tags ──
    gr.HTML(
        '<div class="metro-tags">'
        '<span>CONFORMER</span> <span>CTC</span> <span>KENLM</span> '
        f'<span>CPU OPTIMIZED</span> <span>{_param_count:,} PARAMS</span>'
        '</div>'
    )

    with gr.Tabs():

        # ═══════════════ TAB 1: TRANSCRIBE ═══════════════
        with gr.Tab("📁 Transcribe"):
            audio_input = gr.Audio(
                type="filepath",
                label="Upload or record audio",
                sources=["upload", "microphone"],
                waveform_options=gr.WaveformOptions(
                    waveform_color="#e53935",
                    waveform_progress_color="#ff5252",
                ),
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
                with gr.Row():
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

            output_text = gr.Textbox(
                label="Transcription",
                lines=8,
                elem_classes=["output-text"],
                placeholder="Transcription will appear here...",
            )

            speed_badge = gr.HTML("")
            stats_output = gr.HTML("")

        # ═══════════════ TAB 2: STREAM ═══════════════
        with gr.Tab("🎙 Stream"):
            gr.HTML(
                '<div class="stream-hint">'
                '<span class="pulse-dot"></span>'
                'Speak into your microphone — transcription updates live '
                '(greedy decoding)'
                '</div>'
            )

            stream_state = gr.State(None)

            stream_audio = gr.Audio(
                sources=["microphone"],
                streaming=True,
                type="numpy",
                label="Microphone",
                waveform_options=gr.WaveformOptions(
                    waveform_color="#e53935",
                    waveform_progress_color="#ff5252",
                ),
            )

            stream_output = gr.Textbox(
                label="Transcription",
                lines=8,
                elem_classes=["stream-output"],
                placeholder="Start speaking...",
                interactive=False,
            )

            stream_clear_btn = gr.Button(
                "CLEAR",
                elem_classes=["clear-btn"],
            )

            stream_audio.stream(
                fn=transcribe_streaming,
                inputs=[stream_audio, stream_state],
                outputs=[stream_output, stream_state],
            )

            stream_clear_btn.click(
                fn=clear_stream,
                outputs=[stream_output, stream_state],
            )

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
        share=False,
        theme=METRO_THEME,
        css=CSS,
        ssr_mode=False,
    )
