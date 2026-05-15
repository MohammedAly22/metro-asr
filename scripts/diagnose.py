#!/usr/bin/env python3
"""
Metro-ASR Full Training Diagnostic
Run: python scripts/diagnose.py
"""
import os
import sys
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F

from metro_asr.utils.config import load_config
from metro_asr.model.metro import MetroASR
from metro_asr.model.tokenizer import build_tokenizer
from metro_asr.data.dataset import MetroASRDataset
from metro_asr.data.collator import MetroCollator

CONFIG_PATH = "configs/metro_22m.yaml"
TOKENIZER_DIR = "tokenizer"
PREPARED_DATA_DIR = "data_prepared"

SEP = "=" * 70


def load_batch(config, tokenizer, n=8, augment=False):
    from datasets import load_from_disk
    train_data = load_from_disk(os.path.join(PREPARED_DATA_DIR, "train"))
    dataset = MetroASRDataset(train_data, tokenizer, config, is_training=augment)
    collator = MetroCollator(pad_id=tokenizer.pad_id)
    items = [dataset[i] for i in range(n)]
    return collator(items)


def check_data_pipeline(batch, tokenizer):
    print(f"\n{'[1] DATA PIPELINE':^70}")
    print(SEP)
    feats = batch["features"]
    feat_lens = batch["feature_lengths"]
    toks = batch["tokens"]
    tok_lens = batch["token_lengths"]
    texts = batch["texts"]

    print(f"  Batch size:      {feats.shape[0]}")
    print(f"  Features shape:  {tuple(feats.shape)}")
    print(f"  Feature lengths: {feat_lens.tolist()}")
    print(f"  Token lengths:   {tok_lens.tolist()}")
    print(f"  Token min/max:   {toks.min().item()}/{toks.max().item()}")
    print(f"  Pad ID in batch: {(toks == tokenizer.pad_id).sum().item()} pad tokens")

    print(f"\n  --- Sample texts and round-trip ---")
    for i in range(min(3, len(texts))):
        ids = toks[i, :tok_lens[i]].tolist()
        decoded = tokenizer.decode(ids)
        original = texts[i][:80]
        match = "✅" if decoded.strip() == original.strip() else "❌"
        print(f"  [{i}] text:    '{original}'")
        print(f"      tokens:  {ids[:15]}{'...' if len(ids) > 15 else ''} (len={tok_lens[i].item()})")
        print(f"      decoded: '{decoded[:80]}' {match}")

    print(f"\n  --- Feature statistics (per sample) ---")
    for i in range(min(3, feats.shape[0])):
        f = feats[i, :feat_lens[i]]
        print(f"  [{i}] mean={f.mean():.4f}  std={f.std():.4f}  "
              f"min={f.min():.4f}  max={f.max():.4f}  frames={feat_lens[i].item()}")

    f0 = feats[0, :feat_lens[0]]
    t_var = f0.var(dim=0).mean().item()
    f_var = f0.var(dim=1).mean().item()
    print(f"\n  Temporal variance (across frames): {t_var:.6f} {'✅' if t_var > 0.01 else '⚠️ LOW'}")
    print(f"  Frequency variance (across bins):  {f_var:.6f} {'✅' if f_var > 0.01 else '⚠️ LOW'}")


def check_model_output(model, batch, device):
    print(f"\n{'[2] MODEL OUTPUT ANALYSIS':^70}")
    print(SEP)
    feats = batch["features"].to(device)
    feat_lens = batch["feature_lengths"].to(device)
    tok_lens = batch["token_lengths"]

    model.eval()
    with torch.no_grad():
        log_probs, out_lens, inter_lp = model(feats, feat_lens)

    print(f"  Output shape:    {tuple(log_probs.shape)}  (B, T_out, vocab_size)")
    print(f"  Output lengths:  {out_lens.tolist()}")
    print(f"  Token lengths:   {tok_lens.tolist()}")

    valid = out_lens.cpu() >= tok_lens
    n_invalid = (~valid).sum().item()
    print(f"  CTC valid (out >= target): {valid.sum().item()}/{len(valid)} "
          f"{'✅' if valid.all() else f'⚠️ {n_invalid} INVALID'}")

    if inter_lp:
        print(f"  Intermediate CTC layers: {list(inter_lp.keys())}")

    probs = log_probs.exp().cpu()
    print(f"\n  --- Per-sample output distribution ---")
    max_entropy = math.log(model.vocab_size)

    for i in range(min(3, probs.shape[0])):
        T = out_lens[i].item()
        p = probs[i, :T]

        blank_p = p[:, 0].mean().item()
        top_p = p.max(dim=-1).values.mean().item()
        entropy = -(p * p.clamp(min=1e-10).log()).sum(dim=-1).mean().item()
        pred_ids = p.argmax(dim=-1)
        n_unique = pred_ids.unique().numel()
        mode_id = pred_ids.mode().values.item()
        mode_frac = (pred_ids == mode_id).float().mean().item()

        logit_time_var = log_probs[i, :T].cpu().var(dim=0).mean().item()

        print(f"  [{i}] T={T}")
        print(f"      P(blank) avg:   {blank_p:.6f}  {'⚠️ BLANK SPIKE' if blank_p > 0.5 else '✅'}")
        print(f"      P(top token):   {top_p:.6f}")
        print(f"      Entropy:        {entropy:.2f} / {max_entropy:.2f} "
              f"({entropy / max_entropy * 100:.1f}%)  "
              f"{'⚠️ NEAR UNIFORM' if entropy / max_entropy > 0.95 else '✅'}")
        print(f"      Unique preds:   {n_unique}/{T}")
        print(f"      Most common:    ID={mode_id} ({mode_frac * 100:.1f}% of frames)")
        print(f"      Logit time-var: {logit_time_var:.6f}  "
              f"{'⚠️ COLLAPSED' if logit_time_var < 0.001 else '✅'}")


def check_representation_collapse(model, batch, device):
    print(f"\n{'[3] REPRESENTATION COLLAPSE CHECK':^70}")
    print(SEP)
    feats = batch["features"].to(device)
    feat_lens = batch["feature_lengths"].to(device)

    reps = {}

    def make_hook(name):
        def hook(module, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            reps[name] = o.detach().cpu()
        return hook

    hooks = []
    hooks.append(model.subsampling.register_forward_hook(make_hook("subsampling")))
    for idx in [0, 5, 11]:
        if idx < len(model.encoder.layers):
            hooks.append(model.encoder.layers[idx].register_forward_hook(make_hook(f"layer_{idx}")))

    model.eval()
    with torch.no_grad():
        _, out_lens, _ = model(feats, feat_lens)

    for h in hooks:
        h.remove()

    for name, rep in sorted(reps.items()):
        T = min(out_lens[0].item(), rep.shape[1])
        r = rep[0, :T]

        time_var = r.var(dim=0).mean().item()
        if r.shape[0] > 1:
            cos = F.cosine_similarity(r[:-1], r[1:], dim=-1).mean().item()
        else:
            cos = 0.0
        norm_std = r.norm(dim=-1).std().item()

        collapsed = time_var < 0.001 or cos > 0.99
        tag = "⚠️ COLLAPSED" if collapsed else "✅"
        print(f"  {name:15s}  time_var={time_var:.6f}  cos_sim={cos:.4f}  "
              f"norm_std={norm_std:.4f}  {tag}")


def check_gradient_flow(model, batch, device):
    print(f"\n{'[4] GRADIENT FLOW':^70}")
    print(SEP)
    feats = batch["features"].to(device)
    feat_lens = batch["feature_lengths"].to(device)
    toks = batch["tokens"].to(device)
    tok_lens = batch["token_lengths"].to(device)

    model.train()
    model.zero_grad()

    log_probs, out_lens, inter_lp = model(feats, feat_lens)

    ctc = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    valid = out_lens >= tok_lens
    if not valid.all():
        idx = valid.nonzero(as_tuple=True)[0]
        lp = log_probs[idx]
        ol = out_lens[idx]
        tk = toks[idx]
        tl = tok_lens[idx]
        inter_lp = {k: v[idx] for k, v in inter_lp.items()}
    else:
        lp, ol, tk, tl = log_probs, out_lens, toks, tok_lens

    lp_ctc = lp.float().transpose(0, 1)
    loss = ctc(lp_ctc, tk, ol, tl)

    for k, v in inter_lp.items():
        il = ctc(v.float().transpose(0, 1), tk, ol, tl)
        loss = loss + 0.3 * il

    print(f"  Total loss: {loss.item():.4f}")
    if inter_lp:
        final_only = ctc(lp_ctc, tk, ol, tl).item()
        print(f"  Final CTC:  {final_only:.4f}")
        for k, v in inter_lp.items():
            il = ctc(v.float().transpose(0, 1), tk, ol, tl).item()
            print(f"  Inter CTC (layer {k}): {il:.4f}")

    loss.backward()

    components = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        gn = p.grad.norm().item()
        key = name.split(".")[0]
        if "layers." in name:
            parts = name.split(".")
            li = parts[parts.index("layers") + 1]
            key = f"encoder.layer_{li}"
        elif "intermediate" in name:
            key = "inter_ctc_head"
        elif "ctc_head" in name:
            key = "ctc_head"
        components.setdefault(key, []).append(gn)

    print(f"\n  --- Gradient norms by component ---")
    for comp in sorted(components.keys()):
        norms = components[comp]
        avg = sum(norms) / len(norms)
        mx = max(norms)
        print(f"  {comp:25s}  avg={avg:.6f}  max={mx:.6f}  n={len(norms)}")

    all_norms = [n for ns in components.values() for n in ns]
    zero_g = sum(1 for n in all_norms if n == 0.0)
    nan_g = sum(1 for n in all_norms if math.isnan(n))
    print(f"\n  Zero gradients: {zero_g}/{len(all_norms)} {'⚠️' if zero_g > 0 else '✅'}")
    print(f"  NaN gradients:  {nan_g}/{len(all_norms)} {'⚠️' if nan_g > 0 else '✅'}")


def check_ctc_sanity(model):
    print(f"\n{'[5] CTC LOSS SANITY CHECK':^70}")
    print(SEP)
    V = model.vocab_size
    ctc = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    T, B = 50, 2
    target_lens = torch.tensor([5, 8])
    input_lens = torch.tensor([T, T])
    targets = torch.randint(3, V, (B, 8))

    # Uniform
    uniform_lp = torch.full((T, B, V), -math.log(V))
    ul = ctc(uniform_lp, targets, input_lens, target_lens).item()
    print(f"  Uniform dist loss:     {ul:.4f}  (log(V)={math.log(V):.4f})")

    # All blank
    blank_lp = torch.full((T, B, V), -100.0)
    blank_lp[:, :, 0] = 0.0
    bl = ctc(blank_lp, targets, input_lens, target_lens).item()
    print(f"  All-blank loss:        {bl:.4f}  (should be 0.0 w/ zero_infinity)")

    # Reasonable — correct token at spread positions
    good_lp = torch.full((T, B, V), -10.0)
    good_lp[:, :, 0] = -2.0
    for b in range(B):
        for t_idx in range(target_lens[b].item()):
            pos = t_idx * (T // target_lens[b].item())
            good_lp[pos, b, targets[b, t_idx].item()] = 0.0
    gl = ctc(good_lp, targets, input_lens, target_lens).item()
    print(f"  Correct-at-position:   {gl:.4f}  (should be << uniform)")

    # Check if model's actual loss matches expectation
    print(f"\n  Model vocab_size={V}, log(V)={math.log(V):.4f}")
    print(f"  If model loss ≈ log(V), output is near-uniform → NOT LEARNING")
    print(f"  If model loss << log(V), model is learning token positions")


def check_input_sensitivity(model, batch, device):
    print(f"\n{'[6] INPUT SENSITIVITY CHECK':^70}")
    print(SEP)
    feats = batch["features"].to(device)
    feat_lens = batch["feature_lengths"].to(device)

    model.eval()
    with torch.no_grad():
        lp1, ol1, _ = model(feats, feat_lens)

        # Shuffle features (give different audio)
        perm = torch.randperm(feats.shape[0])
        lp2, ol2, _ = model(feats[perm], feat_lens[perm])

    # Compare outputs for sample 0 vs permuted
    T1 = ol1[0].item()
    T2 = ol2[0].item()
    T = min(T1, T2)

    o1 = lp1[0, :T].cpu()
    o2 = lp2[0, :T].cpu()

    diff = (o1 - o2).abs().mean().item()
    cos = F.cosine_similarity(o1.flatten().unsqueeze(0), o2.flatten().unsqueeze(0)).item()

    print(f"  Output diff (different inputs): {diff:.6f}")
    print(f"  Output cosine sim:              {cos:.4f}")
    if cos > 0.99:
        print(f"  ⚠️ MODEL IGNORES INPUT — same output regardless of audio!")
    else:
        print(f"  ✅ Model produces different outputs for different inputs")


def check_attention_patterns(model, batch, device):
    print(f"\n{'[7] ATTENTION PATTERN ANALYSIS':^70}")
    print(SEP)
    from metro_asr.model.attention import apply_rotary_pos_emb

    feats = batch["features"].to(device)
    feat_lens = batch["feature_lengths"].to(device)

    model.eval()
    with torch.no_grad():
        x, lens = model.subsampling(feats, feat_lens)
        B, T, _ = x.shape
        mask = torch.arange(T, device=device).unsqueeze(0) < lens.unsqueeze(1)

        layer0 = model.encoder.layers[0]
        x = layer0.ff1(x)
        x_n = layer0.attn_norm(x)

        attn = layer0.attn
        q = attn.q_proj(x_n).view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(x_n).view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)

        cos, sin = attn.rotary_emb(T)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        scale = attn.head_dim ** 0.5
        aw = torch.matmul(q, k.transpose(-2, -1)) / scale
        aw = aw.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        aw = torch.softmax(aw, dim=-1)

        T0 = lens[0].item()
        for h in range(min(2, attn.n_heads)):
            a = aw[0, h, :T0, :T0]
            ent = -(a * a.clamp(min=1e-10).log()).sum(-1).mean().item()
            max_ent = math.log(T0)
            diag = a.diag().mean().item()
            top1 = a.max(dim=-1).values.mean().item()

            uniform = ent / max_ent > 0.95
            tag = "⚠️ UNIFORM" if uniform else "✅"
            print(f"  Head {h}: entropy={ent:.2f}/{max_ent:.2f} ({ent/max_ent*100:.1f}%)  "
                  f"diag_avg={diag:.4f}  top1_avg={top1:.4f}  {tag}")


def print_diagnosis(model):
    print(f"\n{'[8] DIAGNOSIS SUMMARY':^70}")
    print(SEP)
    V = model.vocab_size
    print(f"""
  VOCAB SIZE ANALYSIS:
    Current vocab_size = {V}
    log(vocab_size)    = {math.log(V):.2f}

    For CTC, the gradient per token ≈ 1/vocab_size = {1/V:.6f}
    With batch_size=32, the effective gradient signal is very weak.

    Comparison:
      vocab=100  → gradient ≈ 0.01    (char tokenizer)
      vocab=500  → gradient ≈ 0.002   (small BPE)
      vocab=2000 → gradient ≈ 0.0005  (current — 20x weaker than char!)

    RECOMMENDATION: For initial CTC training, use character tokenizer
    (vocab ≈ 110) or reduce BPE vocab to 500. Once the model converges,
    fine-tune with larger BPE vocab.

  EFFECTIVE BATCH SIZE:
    Current: batch_size=32, grad_accum=1 → effective=32
    Recommended: grad_accum=4 → effective=128
    Larger batch reduces gradient noise, critical for CTC convergence.

  LEARNING RATE:
    Current warmup_init_ratio=0.1 → initial LR = 0.0002
    This should be sufficient with the above fixes.
""")


def main():
    print(SEP)
    print(f"{'Metro-ASR Full Training Diagnostic':^70}")
    print(SEP)

    config = load_config(CONFIG_PATH)
    tokenizer = build_tokenizer(config, TOKENIZER_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n  Config:     {CONFIG_PATH}")
    print(f"  Device:     {device}")
    print(f"  Vocab size: {config['tokenizer']['vocab_size']}")
    print(f"  d_model:    {config['model']['d_model']}")
    print(f"  n_layers:   {config['model']['n_layers']}")

    batch = load_batch(config, tokenizer, n=8, augment=False)
    model = MetroASR.from_config(config).to(device)

    print(f"  Parameters: {model.count_parameters():,}")

    check_data_pipeline(batch, tokenizer)
    check_model_output(model, batch, device)
    check_representation_collapse(model, batch, device)
    check_gradient_flow(model, batch, device)
    check_ctc_sanity(model)
    check_input_sensitivity(model, batch, device)
    check_attention_patterns(model, batch, device)
    print_diagnosis(model)

    print(SEP)
    print("Diagnostic complete.")
    print(SEP)


if __name__ == "__main__":
    main()
