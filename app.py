import os
import torch
import torchaudio
import gradio as gr

from metro_asr.utils.config import load_config
from metro_asr.model.metro import MetroASR
from metro_asr.model.tokenizer import build_tokenizer
from metro_asr.data.features import LogMelFeatureExtractor, resample_audio

# ─── Configuration ───────────────────────────────────────────────────────────
CONFIG_PATH = os.environ.get("METRO_CONFIG", "configs/metro_22m.yaml")
TOKENIZER_DIR = os.environ.get("METRO_TOKENIZER_DIR", "tokenizer")
CHECKPOINT_PATH = os.environ.get("METRO_CHECKPOINT", "checkpoints/metro-22m/best_model.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SERVER_PORT = int(os.environ.get("METRO_PORT", 7860))
# ─────────────────────────────────────────────────────────────────────────────

config = load_config(CONFIG_PATH)
tokenizer = build_tokenizer(config, TOKENIZER_DIR)

feature_extractor = LogMelFeatureExtractor(
    sample_rate=config["audio"]["sample_rate"],
    n_mels=config["audio"]["n_mels"],
    n_fft=config["audio"]["n_fft"],
    hop_length=config["audio"]["hop_length"],
    win_length=config["audio"]["win_length"],
)

model = MetroASR.from_config(config)
if os.path.exists(CHECKPOINT_PATH):
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
model = model.to(DEVICE).eval()

param_count = model.count_parameters()
model_name = config["model"]["name"]


def transcribe(audio_path):
    if audio_path is None:
        return "Please upload or record an audio file."

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

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
            log_probs, out_lengths, _ = model(features, feature_lengths)

        decoded_ids = model.decode_greedy(log_probs, out_lengths)
        text = tokenizer.decode(decoded_ids[0])

        return f"{text}\n\n---\nDuration: {duration:.1f}s | Device: {DEVICE}"

    except Exception as e:
        return f"Error: {str(e)}"


with gr.Blocks(
    title="Metro-ASR",
) as demo:
    gr.Markdown(
        f"""
        # Metro-ASR: Egyptian Arabic Speech Recognition
        **Model:** {model_name} | **Parameters:** {param_count:,} | **Device:** {DEVICE}

        Upload an audio file or record from your microphone to get a transcription.
        Optimized for Egyptian Arabic and English code-switching.
        """
    )

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                type="filepath",
                label="Audio Input",
                sources=["upload", "microphone"],
            )
            submit_btn = gr.Button("Transcribe", variant="primary")

        with gr.Column():
            output_text = gr.Textbox(
                label="Transcription",
                lines=6,
            )

    submit_btn.click(fn=transcribe, inputs=audio_input, outputs=output_text)
    audio_input.change(fn=transcribe, inputs=audio_input, outputs=output_text)

    gr.Markdown(
        """
        ---
        **Metro-ASR** - Non-Autoregressive CTC-based ASR for Egyptian Arabic + Code-Switching
        """
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=SERVER_PORT, share=True)
