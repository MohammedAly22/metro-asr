import os
import time
import numpy as np
import torch
import torchaudio
import gradio as gr

from metro_asr.utils.config import load_config
from metro_asr.model.metro import MetroASR
from metro_asr.model.tokenizer import build_tokenizer
from metro_asr.model.decoder import CTCBeamSearchDecoder
from metro_asr.data.features import LogMelFeatureExtractor, resample_audio

# ========================= CONFIGURATION =========================
DEVICE = "cpu"
LM_PATH = "lm/lm_4gram.arpa"
SERVER_PORT = int(os.environ.get("METRO_PORT", 7860))
STREAM_MIN_DURATION = 3.0
STREAM_TARGET_SR = 16000

MODEL_REGISTRY = {
    "Metro-Tiny (26M) BPE": {
        "config": "configs/metro_tiny.yaml",
        "checkpoint": "checkpoints/metro-tiny/best_model.pt",
        "tokenizer_dir": "tokenizer_final",
        "params": "26M",
        "d_model": 256,
        "layers": 12,
        "heads": 4,
        "vocab": 106,
        "status": True,
    },
    "Metro-Small (61M)": {
        "config": "configs/metro_small.yaml",
        "checkpoint": "checkpoints/metro-small-v2/best_model.pt",
        "tokenizer_dir": "tokenizer_bpe5k_v2",
        "params": "58M",
        "d_model": 384,
        "layers": 12,
        "heads": 6,
        "vocab": 5000,
        "status": True,
    },
    "Metro-Medium (238M)": {
        "config": "configs/metro_medium.yaml",
        "checkpoint": "checkpoints/metro-medium/best_model.pt",
        "tokenizer_dir": "tokenizer_final",
        "params": "238M",
        "d_model": 512,
        "layers": 24,
        "heads": 8,
        "vocab": 2000,
        "status": False,
    },
    "Metro-Large (710M)": {
        "config": "configs/metro_large.yaml",
        "checkpoint": "checkpoints/metro-large/best_model.pt",
        "tokenizer_dir": "tokenizer_final",
        "params": "710M",
        "d_model": 768,
        "layers": 32,
        "heads": 12,
        "vocab": 4000,
        "status": False,
    },
}
# =================================================================

# ── Preload default model + LM at startup ──
_default_key = "Metro-Small (61M)"
print(f"Loading {_default_key} model...")
_default_info = MODEL_REGISTRY[_default_key]
_default_config = load_config(_default_info["config"])
_default_tokenizer = build_tokenizer(_default_config, _default_info["tokenizer_dir"])
_default_feature_extractor = LogMelFeatureExtractor(
    sample_rate=_default_config["audio"]["sample_rate"],
    n_mels=_default_config["audio"]["n_mels"],
    n_fft=_default_config["audio"]["n_fft"],
    hop_length=_default_config["audio"]["hop_length"],
    win_length=_default_config["audio"]["win_length"],
)
_default_model = MetroASR.from_config(_default_config)
_ckpt = torch.load(_default_info["checkpoint"], map_location="cpu", weights_only=False)
_default_model.load_state_dict(_ckpt["model_state_dict"])
_default_model = _default_model.to(DEVICE).eval()
_default_param_count = _default_model.count_parameters()
print(f"Model loaded: {_default_param_count:,} params")

_beam_decoder = None
if LM_PATH and os.path.exists(LM_PATH):
    print(f"Loading KenLM from {LM_PATH}...")
    _beam_decoder = CTCBeamSearchDecoder(
        _default_tokenizer,
        lm_path=LM_PATH,
        beam_width=100,
        alpha=0.5,
        beta=5.0,
    )
    print("LM loaded.")

loaded_models = {
    _default_key: (_default_model, _default_tokenizer, _default_feature_extractor, _default_param_count)
}
lm_decoders = {
    _default_info["tokenizer_dir"]: _beam_decoder,
}


def get_available_models():
    available = []
    for name, info in MODEL_REGISTRY.items():
        if info["status"] and os.path.exists(info["checkpoint"]):
            available.append(name)
        elif not info["status"]:
            available.append(f"{name} [Coming Soon]")
        else:
            available.append(f"{name} [No Checkpoint]")
    return available


def load_model(model_key):
    clean_key = model_key.split(" [")[0]

    if clean_key in loaded_models:
        return loaded_models[clean_key]

    if clean_key not in MODEL_REGISTRY:
        return None, None, None, None

    info = MODEL_REGISTRY[clean_key]
    if not info["status"] or not os.path.exists(info["checkpoint"]):
        return None, None, None, None

    config = load_config(info["config"])
    tokenizer = build_tokenizer(config, info["tokenizer_dir"])

    feature_extractor = LogMelFeatureExtractor(
        sample_rate=config["audio"]["sample_rate"],
        n_mels=config["audio"]["n_mels"],
        n_fft=config["audio"]["n_fft"],
        hop_length=config["audio"]["hop_length"],
        win_length=config["audio"]["win_length"],
    )

    model = MetroASR.from_config(config)
    ckpt = torch.load(info["checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE).eval()

    param_count = model.count_parameters()
    loaded_models[clean_key] = (model, tokenizer, feature_extractor, param_count)

    tok_dir = info["tokenizer_dir"]
    if tok_dir not in lm_decoders and LM_PATH and os.path.exists(LM_PATH):
        lm_decoders[tok_dir] = CTCBeamSearchDecoder(
            tokenizer, lm_path=LM_PATH, beam_width=100, alpha=0.5, beta=5.0,
        )

    return model, tokenizer, feature_extractor, param_count


# ── File upload transcription ──

def transcribe(audio_path, model_choice, decoding_method, beam_width, lm_alpha, lm_beta):
    if audio_path is None:
        return "", "", ""

    clean_key = model_choice.split(" [")[0]
    info = MODEL_REGISTRY.get(clean_key)
    if info is None or not info["status"]:
        return "This model is not available yet.", "", ""

    if not os.path.exists(info["checkpoint"]):
        return "Checkpoint not found.", "", ""

    result = load_model(model_choice)
    if result[0] is None:
        return "Failed to load model.", "", ""

    model, tokenizer, feature_extractor, param_count = result

    try:
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0)
        duration = waveform.shape[0] / sr

        if sr != feature_extractor.sample_rate:
            waveform = resample_audio(waveform, sr, feature_extractor.sample_rate)
            if not isinstance(waveform, torch.Tensor):
                waveform = torch.tensor(waveform, dtype=torch.float32)

        features = feature_extractor(waveform)
        features = features.unsqueeze(0).to(DEVICE)
        feature_lengths = torch.tensor([features.shape[1]], dtype=torch.long, device=DEVICE)

        t_start = time.time()
        with torch.no_grad():
            log_probs, out_lengths, _ = model(features, feature_lengths)
        model_time = time.time() - t_start

        use_beam = decoding_method == "Beam Search + LM"

        t_decode_start = time.time()
        if use_beam:
            tok_dir = info["tokenizer_dir"]
            decoder = lm_decoders.get(tok_dir)
            if decoder is not None:
                results = decoder.decode(log_probs, out_lengths)
                text = results[0]
                method_label = f"Beam Search + LM (beam={int(beam_width)}, alpha={lm_alpha}, beta={lm_beta})"
            else:
                decoded_ids = model.decode_greedy(log_probs, out_lengths)
                text = tokenizer.decode(decoded_ids[0])
                method_label = "Greedy (LM not available)"
        else:
            decoded_ids = model.decode_greedy(log_probs, out_lengths)
            text = tokenizer.decode(decoded_ids[0])
            method_label = "Greedy Decoding"

        decode_time = time.time() - t_decode_start
        total_time = time.time() - t_start
        rtf = total_time / duration if duration > 0 else 0

        if rtf < 1.0:
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
            ("Parameters", f"{param_count:,}"),
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

    features = _default_feature_extractor(waveform)
    features = features.unsqueeze(0).to(DEVICE)
    feature_lengths = torch.tensor([features.shape[1]], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        log_probs, out_lengths, _ = _default_model(features, feature_lengths)

    decoded_ids = _default_model.decode_greedy(log_probs, out_lengths)
    text = _default_tokenizer.decode(decoded_ids[0])
    return text.strip()


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
        f'<span>CPU OPTIMIZED</span> <span>{_default_param_count:,} PARAMS</span>'
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

            _available = get_available_models()
            _default_dropdown = next((m for m in _available if m.startswith("Metro-Small")), _available[0])
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=_available,
                    value=_default_dropdown,
                    label="Model",
                    info="Select the model variant",
                )
                decoding_method = gr.Radio(
                    choices=["Greedy", "Beam Search + LM"],
                    value="Beam Search + LM",
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
                f'(using {_default_key}, greedy decoding)'
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
            "**Metro-ASR** is a non-autoregressive CTC-based ASR system "
            "built on a Conformer encoder with: "
            "RoPE, SwiGLU, RMSNorm, SE-Conv, Stochastic Depth, Intermediate CTC. "
            "Trained on 130K+ Egyptian Arabic audio samples with code-switching support. "
            "Achieves **19.90% WER** (beam+LM) on mixed test set at **RTF 0.004** on CPU. "
            "This demo runs entirely on **CPU**.",
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
        inputs=[audio_input, model_dropdown, decoding_method, beam_width, lm_alpha, lm_beta],
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
