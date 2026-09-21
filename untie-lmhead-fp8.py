#!/usr/bin/env python3
"""untie-lmhead-fp8.py -- copy a tied Gemma4 NVFP4 checkpoint into a new dir
with tie_word_embeddings=false and a separately quantized lm_head.weight.

Two modes, chosen from the checkpoint's quantization framework:

  fp8   compressed-tensors NVFP4 checkpoints:
        lm_head.weight -> F8E4M3 [V,H] + lm_head.weight_scale F32 [V,1],
        scheme advertised via a new config_groups["group_1"] targeting
        re:.*lm_head (W8A8 dynamic activations; weight-only FP8 picks the
        HummingFP8 kernel which crashes on ParallelLMHead, see below).

  nvfp4 ModelOpt NVFP4 checkpoints (e.g. the official google/*-NVFP4):
        modelopt_fp4 has no FP8 linear scheme at all -- a non-excluded
        ParallelLMHead can only be NVFP4. Writes the same tensor set the
        other quantized linears carry: lm_head.weight U8 packed e2m1
        [V,H/2], lm_head.weight_scale F8E4M3 [V,H/16], lm_head.weight_scale_2
        F32 scalar (amax/2688), lm_head.input_scale F32 scalar (W4A4-shaped
        placeholder; W4A16 mode discards it after load). Removes "lm_head"
        from hf_quant_config.json quantization.exclude_modules and from
        config.json quantization_config.ignore so the layer is quantized.

Runs inside the tenhkspark/vllm-gb10 image (needs torch + safetensors):

  docker run --rm -v "$HOME/models":/models \
    -v <this dir>:/tools:ro --entrypoint python3 tenhkspark/vllm-gb10:v0.28.0-sm121 \
    /tools/untie-lmhead-fp8.py /models/<src> /models/<dst>

Layout produced:
  - *.safetensors                hardlinked (bytes identical, never modified);
                                 single-file and multi-shard layouts both work
  - lm_head_{fp8,nvfp4}.safetensors  new shard with the lm_head tensors
  - model.safetensors.index.json weight_map + total_size updated
  - config.json                  tie_word_embeddings=false (top + text_config)
"""
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8_MAX = 448.0  # torch.float8_e4m3fn finite max
FP4_MAX = 6.0  # e2m1 finite max
NVFP4_GROUP = 16
EMBED = "model.language_model.embed_tokens.weight"
SKIP_DIRS = {".cache"}

# e2m1 magnitude grid and its round-to-nearest edges.
E2M1_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_EDGES = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])

# NOTE: weight-only FP8 (input_activations=None) resolves to
# CompressedTensorsW8A16Fp8, which on CUDA picks HummingFP8ScaledMMLinearKernel;
# humming_utils.prepare_humming_linear_layer_config reads
# layer.output_partition_sizes, which ParallelLMHead never sets ->
# AttributeError at load (vLLM 0.28.0). W8A8 with dynamic per-token input
# activations resolves to CompressedTensorsW8A8Fp8 ->
# CutlassFP8ScaledMMLinearKernel, which loads and runs on sm121.
LM_HEAD_GROUP = {
    "format": "float-quantized",
    "input_activations": {
        "actorder": None,
        "block_structure": None,
        "dynamic": True,
        "group_size": None,
        "num_bits": 8,
        "observer": None,
        "observer_kwargs": {},
        "scale_dtype": None,
        "strategy": "token",
        "symmetric": True,
        "type": "float",
        "zp_dtype": None,
    },
    "output_activations": None,
    "targets": ["re:.*lm_head"],
    "weights": {
        "actorder": None,
        "block_structure": None,
        "dynamic": False,
        "group_size": None,
        "num_bits": 8,
        "observer": None,
        "observer_kwargs": {},
        "scale_dtype": "torch.float32",
        "strategy": "channel",
        "symmetric": True,
        "type": "float",
        "zp_dtype": None,
    },
}


def detect_mode(src: str) -> str:
    with open(os.path.join(src, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    qc = cfg.get("quantization_config") or {}
    if (qc.get("quant_method") or "").lower() == "modelopt":
        return "nvfp4"
    return "fp8"


def quantize_fp8(w: torch.Tensor):
    """Per-output-channel symmetric FP8 E4M3. Returns {name: tensor}."""
    amax = w.abs().amax(dim=1, keepdim=True)  # [V,1]
    scale = (amax / FP8_MAX).clamp_min(torch.finfo(torch.float32).tiny)
    wq = (w / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    dq = wq.to(torch.float32) * scale
    return {
        "lm_head.weight": wq.contiguous(),
        "lm_head.weight_scale": scale.to(torch.float32).contiguous(),
    }, dq


def quantize_nvfp4(w: torch.Tensor):
    """ModelOpt NVFP4 (W4A16 layout): packed e2m1 + fp8 group scales +
    fp32 global scale = amax/(6*448). Returns ({name: tensor}, dequant)."""
    v, h = w.shape
    assert h % NVFP4_GROUP == 0, f"hidden {h} not multiple of {NVFP4_GROUP}"
    ng = h // NVFP4_GROUP
    amax = w.abs().max()
    s2 = amax / (FP4_MAX * FP8_MAX)  # weight_scale_2, dequant-side global
    xb = w.reshape(v, ng, NVFP4_GROUP)
    bmax = xb.abs().amax(dim=-1)  # [V, ng]
    # block scale stored as fp8; dequant is w = q * s_block * s2, so the
    # ideal block scale is bmax/(FP4_MAX*s2) = bmax*FP8_MAX/amax (<=448).
    s_blk = (bmax * (FP8_MAX / amax)).clamp(max=FP8_MAX)
    s_blk = s_blk.to(torch.float8_e4m3fn)
    s_eff = s_blk.to(torch.float32)
    s_eff = s_eff.clamp_min(torch.finfo(torch.float32).tiny)
    q = xb / (s_eff.unsqueeze(-1) * s2)  # roughly [-6, 6]
    qf = q.reshape(v, h)
    idx = torch.bucketize(qf.abs(), E2M1_EDGES)  # nearest e2m1 level
    dq = (
        E2M1_LEVELS[idx] * torch.sign(qf)
    ).reshape(v, ng, NVFP4_GROUP) * (s_eff.unsqueeze(-1) * s2)
    dq = dq.reshape(v, h)
    # nibble code = level index | sign<<3; low nibble = even element
    codes = idx.to(torch.uint8) | ((qf < 0).to(torch.uint8) << 3)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return {
        "lm_head.weight": packed.contiguous(),
        "lm_head.weight_scale": s_blk.contiguous(),
        "lm_head.weight_scale_2": s2.reshape(()).to(torch.float32),
        "lm_head.input_scale": torch.tensor(1.0, dtype=torch.float32),
    }, dq


def sanity(tag: str, dq: torch.Tensor, w: torch.Tensor) -> None:
    d64, w64 = dq.to(torch.float64), w.to(torch.float64)
    rel = ((d64 - w64).norm() / w64.norm()).item()
    cos = torch.nn.functional.cosine_similarity(
        d64.flatten(), w64.flatten(), dim=0
    ).item()
    print(f"{tag} sanity: rel_fro_err={rel:.6f} cos={cos:.8f}", file=sys.stderr)


def drop_exact(lst, name):
    return [x for x in lst if x != name]


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    assert os.path.isdir(src), src
    mode = detect_mode(src)
    shard = f"lm_head_{mode}.safetensors"
    os.makedirs(dst, exist_ok=False)  # dst must be new; never overwrite
    print(f"mode={mode} dst={dst}", file=sys.stderr)

    # ---- 0. weight_map: which shard holds the tied embedding ----
    idx_src = os.path.join(src, "model.safetensors.index.json")
    weight_map = None
    if os.path.exists(idx_src):
        with open(idx_src, encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]

    # ---- 1. read the tied embedding, quantize a copy ----
    emb_file = (
        weight_map.get(EMBED)
        if weight_map is not None
        else "model.safetensors"
    )
    assert emb_file, f"{EMBED} not in weight_map"
    with safe_open(
        os.path.join(src, emb_file), framework="pt", device="cpu"
    ) as f:
        emb = f.get_tensor(EMBED)  # [vocab, hidden] bf16
    print(
        f"embed {tuple(emb.shape)} {emb.dtype} from {emb_file}",
        file=sys.stderr,
    )

    w = emb.to(torch.float32)
    tensors, dq = (quantize_nvfp4 if mode == "nvfp4" else quantize_fp8)(w)
    sanity(mode, dq, w)
    del dq, w, emb

    # ---- 2. copy the tree; hardlink the big safetensors ----
    for name in os.listdir(src):
        if name in SKIP_DIRS:
            continue
        s, d = os.path.join(src, name), os.path.join(dst, name)
        if os.path.isdir(s):
            shutil.copytree(s, d)
        elif name.endswith(".safetensors"):
            os.link(s, d)  # same inode; we never write to it
        else:
            shutil.copy2(s, d)

    # ---- 3. write the lm_head shard ----
    save_file(tensors, os.path.join(dst, shard), metadata={"format": "pt"})
    shard_bytes = os.path.getsize(os.path.join(dst, shard))
    print(f"wrote {shard}: {shard_bytes / 2**30:.3f} GiB", file=sys.stderr)

    # ---- 4. index (single-file checkpoints may have none) ----
    idx_path = os.path.join(dst, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        with open(idx_path, encoding="utf-8") as f:
            idx = json.load(f)
        for name in tensors:
            idx["weight_map"][name] = shard
        idx["metadata"]["total_size"] += shard_bytes
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(idx, f)

    # ---- 5. config.json: untie (+ lm_head scheme for fp8 mode) ----
    cfg_path = os.path.join(dst, "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["tie_word_embeddings"] = False
    if isinstance(cfg.get("text_config"), dict):
        cfg["text_config"]["tie_word_embeddings"] = False
    qc = cfg.get("quantization_config") or {}
    if mode == "fp8":
        qc["config_groups"]["group_1"] = LM_HEAD_GROUP
    else:
        # modelopt: un-exclude lm_head so it gets the NVFP4 linear method
        if isinstance(qc.get("ignore"), list):
            qc["ignore"] = drop_exact(qc["ignore"], "lm_head")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")

    # ---- 6. nvfp4 mode: un-exclude lm_head in hf_quant_config.json ----
    if mode == "nvfp4":
        hfq_path = os.path.join(dst, "hf_quant_config.json")
        if os.path.exists(hfq_path):
            with open(hfq_path, encoding="utf-8") as f:
                hfq = json.load(f)
            q = hfq.get("quantization") or {}
            if isinstance(q.get("exclude_modules"), list):
                q["exclude_modules"] = drop_exact(
                    q["exclude_modules"], "lm_head"
                )
            with open(hfq_path, "w", encoding="utf-8") as f:
                json.dump(hfq, f, indent=2)
                f.write("\n")

    print("done:", dst, file=sys.stderr)


if __name__ == "__main__":
    main()
