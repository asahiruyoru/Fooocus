#!/usr/bin/env python3
"""
Worker-ish Anima Preview2 reproduction path.

This follows Fooocus' default_pipeline path more closely than the direct
test_anima_pipeline sampler. It is useful when we want to compare:

- bare prompts such as "1girl"
- richer prompts
- direct sampler path vs worker-ish path

Typical Colab usage:

    cd /content/Fooocus
    python scripts/anima_preview2_worker_repro.py \
        --label worker_short_20_1024 \
        --prompt "1girl"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.request import urlretrieve


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path("/content/anima_case_outputs")
FOOOCUS_EXPANSION_BIN_URL = (
    "https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin"
)


def ensure_repo_root() -> None:
    os.chdir(REPO_ROOT)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    sys.argv = ["fooocus", "--preset", "anima_preview2"]


def ensure_expansion_model() -> Path:
    import modules.config

    expansion_dir = Path(modules.config.path_fooocus_expansion)
    expansion_dir.mkdir(parents=True, exist_ok=True)
    expansion_bin = expansion_dir / "pytorch_model.bin"

    if expansion_bin.exists() and expansion_bin.stat().st_size > 0:
        return expansion_bin

    print(f"Downloading Fooocus expansion model: {FOOOCUS_EXPANSION_BIN_URL}")
    urlretrieve(FOOOCUS_EXPANSION_BIN_URL, expansion_bin)
    return expansion_bin


def compute_image_metrics(image_path: Path) -> dict[str, float]:
    import numpy as np
    from PIL import Image

    arr = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)
    vertical = float(abs(gray[1:] - gray[:-1]).mean()) if gray.shape[0] > 1 else 0.0
    horizontal = float(abs(gray[:, 1:] - gray[:, :-1]).mean()) if gray.shape[1] > 1 else 0.0
    return {
        "hf_metric": (vertical + horizontal) / 2.0,
        "color_std": float(arr.std()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Worker-ish default_pipeline repro for Anima Preview2."
    )
    parser.add_argument("--label", default="worker_case", help="Case label.")
    parser.add_argument("--prompt", default="1girl", help="Positive prompt.")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt.")
    parser.add_argument("--steps", type=int, default=20, help="Sampling steps.")
    parser.add_argument("--width", type=int, default=1024, help="Width.")
    parser.add_argument("--height", type=int, default=1024, help="Height.")
    parser.add_argument("--cfg", type=float, default=4.0, help="CFG scale.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument(
        "--switch",
        type=int,
        default=-1,
        help="Refiner switch step. Defaults to steps // 2.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path. Defaults to <output-root>/<label>.png",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Output directory when --output is omitted.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_repo_root()
    ensure_expansion_model()

    import modules.config
    import modules.default_pipeline as pipeline
    import modules.patch
    import numpy as np
    from PIL import Image
    from modules.patch import PatchSettings

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else output_root / f"{args.label}.png"
    switch = args.switch if args.switch >= 0 else max(args.steps // 2, 1)

    pid = os.getpid()
    modules.patch.patch_settings[pid] = PatchSettings(
        sharpness=modules.config.default_sample_sharpness,
        adm_scaler_end=0.3,
        positive_adm_scale=1.5,
        negative_adm_scale=0.8,
        controlnet_softness=0.25,
        adaptive_cfg=modules.config.default_cfg_tsnr,
    )

    pipeline.refresh_everything(
        refiner_model_name="None",
        base_model_name="anima-preview2.safetensors",
        loras=[],
        base_model_additional_loras=[],
        use_synthetic_refiner=False,
        vae_name="qwen_image_vae.safetensors",
    )

    started = time.time()
    positive = pipeline.clip_encode([args.prompt], pool_top_k=1)
    negative = pipeline.clip_encode([args.negative_prompt], pool_top_k=1)
    imgs = pipeline.process_diffusion(
        positive_cond=positive,
        negative_cond=negative,
        steps=args.steps,
        switch=switch,
        width=args.width,
        height=args.height,
        image_seed=args.seed,
        callback=lambda *unused_args, **unused_kwargs: None,
        sampler_name="euler",
        scheduler_name="simple",
        latent=None,
        denoise=1.0,
        tiled=False,
        cfg_scale=args.cfg,
        refiner_swap_method="joint",
        disable_preview=True,
    )

    image_array = imgs[0]
    Image.fromarray(image_array).save(output_path)

    metrics = compute_image_metrics(output_path)
    result = {
        "label": args.label,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "path_type": "workerish_default_pipeline",
        "steps": args.steps,
        "width": args.width,
        "height": args.height,
        "cfg": args.cfg,
        "seed": args.seed,
        "status": "ok",
        "output_path": str(output_path),
        "duration_sec": round(time.time() - started, 2),
        "hf_metric": metrics["hf_metric"],
        "color_std": metrics["color_std"],
    }

    print("RESULT", json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
