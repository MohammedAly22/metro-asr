import os
import copy
import yaml


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    defaults = {
        "model": {
            "name": "metro-asr",
            "d_model": 256,
            "n_layers": 12,
            "n_heads": 4,
            "ff_multiplier": 3,
            "conv_kernel_size": 31,
            "dropout": 0.1,
            "conv_expansion_factor": 2,
            "se_ratio": 4,
            "stochastic_depth_rate": 0.0,
        },
        "audio": {
            "sample_rate": 16000,
            "n_mels": 80,
            "n_fft": 512,
            "hop_length": 160,
            "win_length": 400,
        },
        "tokenizer": {
            "type": "bpe",
            "vocab_size": 5000,
        },
        "training": {
            "batch_size": 32,
            "grad_accumulation_steps": 1,
            "learning_rate": 0.001,
            "warmup_steps": 10000,
            "max_steps": 500000,
            "weight_decay": 0.01,
            "max_grad_norm": 5.0,
            "fp16": True,
            "ctc_loss_reduction": "mean",
            "intermediate_ctc_layers": [],
            "intermediate_ctc_weight": 0.3,
            "max_audio_duration": 30.0,
            "min_audio_duration": 0.5,
            "spec_augment": True,
            "freq_mask_param": 27,
            "time_mask_param": 100,
            "n_freq_masks": 2,
            "n_time_masks": 10,
            "speed_perturb": True,
            "speed_perturb_factors": [0.9, 1.0, 1.1],
            "save_every_n_steps": 5000,
            "eval_every_n_steps": 1000,
            "log_every_n_steps": 10,
            "checkpoint_dir": "checkpoints",
            "resume_from": None,
            "wandb_project": "metro-asr",
            "wandb_run_name": None,
        },
        "data": {
            "datasets": [],
            "eval_split_ratio": 0.02,
            "num_workers": 4,
            "cache_dir": "data_cache",
        },
    }

    merged = _deep_merge(defaults, config)

    env_overrides = {
        "METRO_BATCH_SIZE": ("training", "batch_size", int),
        "METRO_LR": ("training", "learning_rate", float),
        "METRO_MAX_STEPS": ("training", "max_steps", int),
        "METRO_FP16": ("training", "fp16", lambda x: x.lower() == "true"),
        "METRO_WORKERS": ("data", "num_workers", int),
    }

    for env_key, (section, key, cast) in env_overrides.items():
        val = os.environ.get(env_key)
        if val is not None:
            merged[section][key] = cast(val)

    return merged


def _deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result
