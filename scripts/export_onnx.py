import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metro_asr.utils.config import load_config
from metro_asr.utils.logger import get_logger, print_banner
from metro_asr.model.metro import MetroASR
from metro_asr.model.tokenizer import build_tokenizer

# ─── Configuration ───────────────────────────────────────────────────────────
CONFIG_PATH = "configs/metro_22m.yaml"
TOKENIZER_DIR = "tokenizer"
CHECKPOINT_PATH = "checkpoints/metro-22m/best_model.pt"
OUTPUT_PATH = "exports/metro_22m.onnx"
OPSET_VERSION = 17
MAX_AUDIO_FRAMES = 3000
# ─────────────────────────────────────────────────────────────────────────────


class MetroASRONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features, feature_lengths):
        log_probs, lengths, _ = self.model(features, feature_lengths)
        return log_probs, lengths


def main():
    logger = get_logger("metro-asr")
    print_banner()

    config = load_config(CONFIG_PATH)
    build_tokenizer(config, TOKENIZER_DIR)

    model = MetroASR.from_config(config)

    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"📦 Loaded checkpoint: {CHECKPOINT_PATH}")
    else:
        logger.warning(f"⚠️  No checkpoint, exporting random weights")

    model.eval()
    wrapper = MetroASRONNXWrapper(model)

    n_mels = config["audio"]["n_mels"]
    dummy_features = torch.randn(1, MAX_AUDIO_FRAMES, n_mels)
    dummy_lengths = torch.tensor([MAX_AUDIO_FRAMES], dtype=torch.long)

    os.makedirs(os.path.dirname(OUTPUT_PATH) if os.path.dirname(OUTPUT_PATH) else ".", exist_ok=True)

    logger.info(f"📦 Exporting ONNX model to {OUTPUT_PATH}...")

    torch.onnx.export(
        wrapper,
        (dummy_features, dummy_lengths),
        OUTPUT_PATH,
        opset_version=OPSET_VERSION,
        input_names=["features", "feature_lengths"],
        output_names=["log_probs", "output_lengths"],
        dynamic_axes={
            "features": {0: "batch", 1: "time"},
            "feature_lengths": {0: "batch"},
            "log_probs": {0: "batch", 1: "time"},
            "output_lengths": {0: "batch"},
        },
    )

    file_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    logger.info(f"✅ ONNX export complete: {file_size:.1f} MB")

    try:
        import onnx
        onnx_model = onnx.load(OUTPUT_PATH)
        onnx.checker.check_model(onnx_model)
        logger.info("✅ ONNX model validation passed")
    except ImportError:
        logger.warning("⚠️  onnx package not installed, skipping validation")
    except Exception as e:
        logger.error(f"❌ ONNX validation failed: {e}")


if __name__ == "__main__":
    main()
