import os
import time
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
TOKENIZER_DIR = "tokenizer_final"
LM_PATH = "lm/lm_4gram.arpa"
SERVER_PORT = int(os.environ.get("METRO_PORT", 7860))

MODEL_REGISTRY = {
    "Metro-Tiny (26M)": {
        "config": "configs/metro_tiny.yaml",
        "checkpoint": "checkpoints/metro-tiny/best_model.pt",
        "params": "26M",
        "d_model": 256,
        "layers": 12,
        "heads": 4,
        "vocab": 600,
        "status": True,
    },
    "Metro-Small (58M)": {
        "config": "configs/metro_small.yaml",
        "checkpoint": "checkpoints/metro-small/best_model.pt",
        "params": "58M",
        "d_model": 384,
        "layers": 12,
        "heads": 6,
        "vocab": 1000,
        "status": False,
    },
    "Metro-Medium (238M)": {
        "config": "configs/metro_medium.yaml",
        "checkpoint": "checkpoints/metro-medium/best_model.pt",
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
        "params": "710M",
        "d_model": 768,
        "layers": 32,
        "heads": 12,
        "vocab": 4000,
        "status": False,
    },
    "Metro-XLarge (1.4B)": {
        "config": "configs/metro_xlarge.yaml",
        "checkpoint": "checkpoints/metro-xlarge/best_model.pt",
        "params": "1.4B",
        "d_model": 1024,
        "layers": 36,
        "heads": 16,
        "vocab": 8000,
        "status": False,
    },
}
# =================================================================

loaded_models = {}


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
    tokenizer = build_tokenizer(config, TOKENIZER_DIR)

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
    return model, tokenizer, feature_extractor, param_count


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
        if use_beam and os.path.exists(LM_PATH):
            decoder = CTCBeamSearchDecoder(
                tokenizer,
                lm_path=LM_PATH,
                beam_width=int(beam_width),
                alpha=lm_alpha,
                beta=lm_beta,
            )
            results = decoder.decode(log_probs, out_lengths)
            text = results[0]
            method_label = f"Beam Search + LM (beam={int(beam_width)}, α={lm_alpha}, β={lm_beta})"
        else:
            decoded_ids = model.decode_greedy(log_probs, out_lengths)
            text = tokenizer.decode(decoded_ids[0])
            method_label = "Greedy Decoding"

        decode_time = time.time() - t_decode_start
        total_time = time.time() - t_start
        rtf = total_time / duration if duration > 0 else 0

        stats = (
            f"| Metric | Value |\n"
            f"|---|---|\n"
            f"| Audio Duration | {duration:.2f}s |\n"
            f"| Model Inference | {model_time*1000:.1f}ms |\n"
            f"| Decoding Time | {decode_time*1000:.1f}ms |\n"
            f"| Total Latency | {total_time*1000:.1f}ms |\n"
            f"| RTF (Real-Time Factor) | {rtf:.4f} |\n"
            f"| Decoding Method | {method_label} |\n"
            f"| Device | {DEVICE.upper()} |\n"
            f"| Parameters | {param_count:,} |"
        )

        if rtf < 1.0:
            speed_note = f"**{1/rtf:.1f}x** faster than real-time"
        else:
            speed_note = f"**{rtf:.1f}x** slower than real-time"

        return text, stats, speed_note

    except Exception as e:
        return f"Error: {str(e)}", "", ""


def on_decoding_change(method):
    is_beam = method == "Beam Search + LM"
    return (
        gr.update(visible=is_beam),
        gr.update(visible=is_beam),
    )


CSS = """
.main-header {
    text-align: center !important;
    padding: 20px 0 5px 0 !important;
}
.main-header h1 {
    font-size: 2.6em !important;
    font-weight: 800 !important;
    color: #c0392b !important;
    letter-spacing: -0.5px !important;
}
.subtitle {
    text-align: center !important;
    color: #666 !important;
    font-size: 1.05em !important;
    margin-top: -10px !important;
    margin-bottom: 20px !important;
}
.output-text textarea {
    font-size: 1.25em !important;
    line-height: 1.8 !important;
    direction: rtl !important;
    text-align: right !important;
    font-family: 'Noto Sans Arabic', 'Segoe UI', Tahoma, sans-serif !important;
    min-height: 120px !important;
}
.speed-badge {
    text-align: center !important;
    font-size: 1.05em !important;
    padding: 6px 0 !important;
}
.beam-section {
    border: 1px solid #e0e0e0 !important;
    border-radius: 10px !important;
    padding: 14px !important;
    background: #fafafa !important;
    margin-top: 4px !important;
}
.about-section {
    background: #fdf6f5 !important;
    border: 1px solid #f0d0cc !important;
    border-radius: 12px !important;
    padding: 18px !important;
}
footer {visibility: hidden !important;}
"""

METRO_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.red,
    secondary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
).set(
    button_primary_background_fill="linear-gradient(135deg, #c0392b 0%, #e74c3c 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #a93226 0%, #c0392b 100%)",
    button_primary_text_color="#ffffff",
    block_title_text_color="#2c3e50",
    block_label_text_color="#2c3e50",
    checkbox_label_text_color="#2c3e50",
    input_border_color_focus="#e74c3c",
    slider_color="#c0392b",
)

with gr.Blocks(title="Metro-ASR") as demo:

    gr.Markdown("# Metro-ASR", elem_classes=["main-header"])
    gr.Markdown(
        "Non-Autoregressive CTC-based Speech Recognition for Egyptian Arabic + English Code-Switching",
        elem_classes=["subtitle"],
    )

    with gr.Row(equal_height=True):
        # ── Left: Controls ──
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=get_available_models(),
                value=get_available_models()[0],
                label="Model Size",
                info="Select the model variant to use for transcription",
            )

            audio_input = gr.Audio(
                type="filepath",
                label="Audio Input",
                sources=["upload", "microphone"],
            )

            decoding_method = gr.Radio(
                choices=["Greedy", "Beam Search + LM"],
                value="Greedy",
                label="Decoding Method",
                info="Greedy is faster; Beam Search + LM is more accurate",
            )

            with gr.Group(visible=False, elem_classes=["beam-section"]) as beam_group:
                gr.Markdown("**Beam Search Parameters**")
                beam_width = gr.Slider(
                    minimum=5, maximum=500, value=100, step=5,
                    label="Beam Width",
                    info="Number of parallel hypotheses. Higher = more accurate but slower",
                )
                lm_alpha = gr.Slider(
                    minimum=0.0, maximum=3.0, value=0.5, step=0.1,
                    label="LM Weight (Alpha)",
                    info="Language model influence. Higher values favor linguistically valid words",
                )
                lm_beta = gr.Slider(
                    minimum=0.0, maximum=10.0, value=5.0, step=0.5,
                    label="Word Insertion Bonus (Beta)",
                    info="Prevents word deletion. Increase if English words are being dropped",
                )
                gr.Markdown(
                    "*Uses a 4-gram KenLM trained on Egyptian Arabic + English text*"
                )

            submit_btn = gr.Button(
                "Transcribe",
                variant="primary",
                size="lg",
            )

        # ── Right: Output ──
        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="Transcription",
                lines=5,
                elem_classes=["output-text"],
                placeholder="Transcription will appear here...",
            )

            speed_badge = gr.Markdown("", elem_classes=["speed-badge"])

            stats_output = gr.Markdown("")

    # ── About ──
    with gr.Accordion("About Metro-ASR", open=False):
        gr.Markdown(
            "**Metro-ASR** is a non-autoregressive CTC-based speech recognition system "
            "built on a Conformer encoder with modern components:\n\n"
            "- **RoPE** for relative position encoding\n"
            "- **SwiGLU** activation in feed-forward layers\n"
            "- **RMSNorm** for stable training\n"
            "- **SE-Conv** for channel attention\n"
            "- **Stochastic Depth** for regularization\n"
            "- **Intermediate CTC** loss for better gradient flow\n"
            "- **BPE Tokenizer** trained on Egyptian Arabic + English\n\n"
            "Trained on 130k+ audio samples from multiple Egyptian Arabic datasets "
            "including TTS, emotional speech, broadcast data, and a dedicated "
            "Arabic-English code-switching dataset.\n\n"
            "This demo runs on **CPU** to demonstrate model efficiency.",
            elem_classes=["about-section"],
        )

    gr.Markdown(
        "<center style='color:#999; font-size:0.85em; padding:10px 0;'>"
        "Metro-ASR — Conformer + CTC + KenLM | CPU-Optimized"
        "</center>"
    )

    # ── Events ──
    decoding_method.change(
        fn=on_decoding_change,
        inputs=[decoding_method],
        outputs=[beam_group, beam_group],
    )

    submit_btn.click(
        fn=transcribe,
        inputs=[audio_input, model_dropdown, decoding_method, beam_width, lm_alpha, lm_beta],
        outputs=[output_text, stats_output, speed_badge],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=SERVER_PORT, share=False, theme=METRO_THEME, css=CSS, ssr_mode=False)
