#!/usr/bin/env python3
"""
One-command Colab helper for a fresh Anima Preview2 investigation runtime.

Typical usage on Colab after cloning the repo:

    cd /content/Fooocus
    python scripts/anima_preview2_colab_bootstrap.py

This script is intentionally Colab-friendly:
- installs Python requirements from requirements_versions.txt
- downloads the required Anima model files
- runs the plain headless repro cases that previously produced plain_*.png
- writes images and compare_results.json under /content/anima_case_outputs

You can also use it for a single custom case:

    python scripts/anima_preview2_colab_bootstrap.py \
        --profile single \
        --label plain_custom \
        --prompt "1girl" \
        --steps 20 \
        --width 1024 \
        --height 1024
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlretrieve


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path("/content/anima_case_outputs")
DEFAULT_COMFYUI_ROOT = Path("/content/ComfyUI")
BUNDLED_COMFYUI_ROOT = REPO_ROOT / "comfyui_tmp"
COMFYUI_REPO_URL = "https://github.com/comfyanonymous/ComfyUI.git"
REQUIREMENTS_FILE = REPO_ROOT / "requirements_versions.txt"
PRESET_FILE = REPO_ROOT / "presets" / "anima_preview2.json"
PIPELINE_TEST_FILE = REPO_ROOT / "tests" / "test_anima_pipeline.py"
WORKER_REPRO_FILE = REPO_ROOT / "scripts" / "anima_preview2_worker_repro.py"
REQUIRED_REPO_FILES = [
    PRESET_FILE,
    PIPELINE_TEST_FILE,
    WORKER_REPRO_FILE,
]

# The Anima preset tracks checkpoint + VAE downloads, but not the text encoder.
CLIP_DOWNLOADS = {
    "qwen_3_06b_base.safetensors": (
        "https://huggingface.co/circlestone-labs/Anima/resolve/main/"
        "split_files/text_encoders/qwen_3_06b_base.safetensors"
    )
}

DEFAULT_CASES = [
    {
        "label": "plain_short_20_1024",
        "prompt": "1girl",
        "steps": 20,
        "width": 1024,
        "height": 1024,
        "cfg": 4.0,
        "seed": 42,
    },
    {
        "label": "plain_short_60_1024",
        "prompt": "1girl",
        "steps": 60,
        "width": 1024,
        "height": 1024,
        "cfg": 4.0,
        "seed": 42,
    },
    {
        "label": "plain_short_60_1344",
        "prompt": "1girl",
        "steps": 60,
        "width": 1344,
        "height": 1344,
        "cfg": 4.0,
        "seed": 42,
    },
]

RICH_PROMPT = (
    "1girl, solo, anime style, detailed face, long black hair, blue eyes, "
    "school uniform, cherry blossom"
)

DEFAULT_PRESET_PROMPT = (
    "masterpiece, best quality, highres, safe, 1girl, solo, looking at "
    "viewer, smile, long hair, detailed eyes, detailed face, clean lines, "
    "smooth shading, soft lighting"
)

DEFAULT_PRESET_NEGATIVE_PROMPT = (
    "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg "
    "artifacts, watermark, patreon logo, bad hands, bad fingers, bad eyes, "
    "bad pupils, bad iris, 6 fingers, 6 toes"
)

OFFICIAL_TAG_PROMPT = (
    "masterpiece, best quality, highres, safe, 1girl, solo, long black hair, "
    "blue eyes, school uniform, cherry blossom, looking at viewer, blush, "
    "long hair, detailed face, clean lines, smooth shading"
)

OFFICIAL_NEGATIVE_PROMPT = DEFAULT_PRESET_NEGATIVE_PROMPT

WORKER_PAIR_CASES = [
    {
        "runner": "worker",
        "label": "worker_short_20_1024",
        "prompt": "1girl",
        "negative_prompt": "",
        "steps": 20,
        "width": 1024,
        "height": 1024,
        "cfg": 4.0,
        "seed": 42,
    },
    {
        "runner": "worker",
        "label": "worker_rich_20_1024",
        "prompt": RICH_PROMPT,
        "negative_prompt": "",
        "steps": 20,
        "width": 1024,
        "height": 1024,
        "cfg": 4.0,
        "seed": 42,
    },
]

DIAGNOSTIC_CASES = [
    {
        "label": "plain_short_20_1024",
        "prompt": "1girl",
        "steps": 20,
        "width": 1024,
        "height": 1024,
        "cfg": 4.0,
        "seed": 42,
    },
    {
        "label": "plain_short_20_1024_negempty",
        "prompt": "1girl",
        "negative_prompt": "",
        "negative_mode": "encode",
        "steps": 20,
        "width": 1024,
        "height": 1024,
        "cfg": 4.0,
        "seed": 42,
    },
    {
        "label": "plain_rich_20_1024",
        "prompt": RICH_PROMPT,
        "steps": 20,
        "width": 1024,
        "height": 1024,
        "cfg": 4.0,
        "seed": 42,
    },
    *WORKER_PAIR_CASES,
]

OFFICIAL_CASES = [
    {
        "label": "plain_official_euler_a_40_1024",
        "prompt": OFFICIAL_TAG_PROMPT,
        "negative_prompt": OFFICIAL_NEGATIVE_PROMPT,
        "negative_mode": "encode",
        "steps": 40,
        "width": 1024,
        "height": 1024,
        "cfg": 4.5,
        "seed": 42,
        "sampler": "euler_ancestral",
        "scheduler": "simple",
    },
    {
        "label": "plain_official_dpmpp2m_40_1024",
        "prompt": OFFICIAL_TAG_PROMPT,
        "negative_prompt": OFFICIAL_NEGATIVE_PROMPT,
        "negative_mode": "encode",
        "steps": 40,
        "width": 1024,
        "height": 1024,
        "cfg": 4.5,
        "seed": 42,
        "sampler": "dpmpp_2m_sde_gpu",
        "scheduler": "karras",
    },
    {
        "runner": "worker",
        "label": "worker_official_euler_a_40_1024",
        "prompt": OFFICIAL_TAG_PROMPT,
        "negative_prompt": OFFICIAL_NEGATIVE_PROMPT,
        "steps": 40,
        "width": 1024,
        "height": 1024,
        "cfg": 4.5,
        "seed": 42,
        "sampler": "euler_ancestral",
        "scheduler": "simple",
    },
    {
        "runner": "worker",
        "label": "worker_official_dpmpp2m_40_1024",
        "prompt": OFFICIAL_TAG_PROMPT,
        "negative_prompt": OFFICIAL_NEGATIVE_PROMPT,
        "steps": 40,
        "width": 1024,
        "height": 1024,
        "cfg": 4.5,
        "seed": 42,
        "sampler": "dpmpp_2m_sde_gpu",
        "scheduler": "karras",
    },
]


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("$", shlex.join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def run_capture(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    print("$", shlex.join(cmd))
    completed = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    if completed.stdout:
        lines = completed.stdout.splitlines()
        if len(lines) > 80:
            print(f"[truncated output] showing last 80 of {len(lines)} lines")
            lines = lines[-80:]
        text = "\n".join(lines)
        print(text, end="" if text.endswith("\n") else "\n")
    return completed


def repo_file(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


def ensure_repo_root() -> None:
    os.chdir(REPO_ROOT)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def ensure_required_repo_files() -> None:
    missing = [path for path in REQUIRED_REPO_FILES if not path.exists()]
    if not missing:
        return

    missing_text = "\n".join(f"  - {path}" for path in missing)
    raise SystemExit(
        "This checkout is missing files required for the Anima Preview2 Colab bootstrap.\n"
        f"{missing_text}\n"
        "Update or reclone feature/anima-preview2-integration before rerunning."
    )


def show_runtime_info() -> None:
    print("=" * 60)
    print("Anima Preview2 Colab Bootstrap")
    print("=" * 60)
    print("repo_root =", REPO_ROOT)
    print("python =", sys.executable)
    print("cwd =", Path.cwd())
    try:
        import torch

        print("torch.cuda.is_available =", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("device =", torch.cuda.get_device_name(0))
    except Exception as exc:  # pragma: no cover - informational only
        print("torch inspection failed:", repr(exc))


def ensure_python_requirements(skip: bool) -> None:
    if skip:
        print("Skipping Python requirements installation.")
        return

    run([sys.executable, "-m", "pip", "install", "-q", "-r", str(REQUIREMENTS_FILE)], cwd=REPO_ROOT)


def load_anima_download_plan() -> list[tuple[Path, str]]:
    with PRESET_FILE.open(encoding="utf-8") as handle:
        preset = json.load(handle)

    plan = []
    for file_name, url in preset.get("checkpoint_downloads", {}).items():
        plan.append((repo_file("models", "checkpoints", file_name), url))
    for file_name, url in CLIP_DOWNLOADS.items():
        plan.append((repo_file("models", "clip", file_name), url))
    for file_name, url in preset.get("vae_downloads", {}).items():
        plan.append((repo_file("models", "vae", file_name), url))
    return plan


def download_with_python(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading with urllib: {url}")
    urlretrieve(url, destination)


def ensure_models(skip: bool) -> None:
    if skip:
        print("Skipping model download.")
        return

    for destination, url in load_anima_download_plan():
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.stat().st_size > 0:
            size_gb = destination.stat().st_size / 1024 ** 3
            print(f"Already present: {destination} ({size_gb:.2f} GB)")
            continue

        if shutil.which("wget"):
            run(["wget", "-nv", "-O", str(destination), url], cwd=destination.parent)
        else:  # pragma: no cover - Colab normally has wget
            download_with_python(url, destination)

    print("\nModel locations:")
    for destination, _ in load_anima_download_plan():
        size_gb = destination.stat().st_size / 1024 ** 3 if destination.exists() else 0
        print(f"  {destination}: {size_gb:.2f} GB")


def _comfy_root_has_required_modules(comfy_root: Path) -> bool:
    comfy_sd = comfy_root / "comfy" / "sd.py"
    if not comfy_sd.exists():
        return False

    root_str = str(comfy_root)
    added = False
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
        added = True

    try:
        return importlib.util.find_spec("comfy_aimdo.host_buffer") is not None
    finally:
        if added:
            try:
                sys.path.remove(root_str)
            except ValueError:
                pass


def ensure_comfyui_reference() -> Path:
    candidate_roots = [BUNDLED_COMFYUI_ROOT, DEFAULT_COMFYUI_ROOT]

    for comfy_root in candidate_roots:
        if _comfy_root_has_required_modules(comfy_root):
            source = "bundled" if comfy_root == BUNDLED_COMFYUI_ROOT else "existing"
            print(f"Using {source} ComfyUI reference checkout: {comfy_root}")
            os.environ["FOOOCUS_ANIMA_COMFY_ROOT"] = str(comfy_root)
            os.environ["ANIMA_COMFY_ROOT"] = str(comfy_root)
            print(f"FOOOCUS_ANIMA_COMFY_ROOT={comfy_root}")
            return comfy_root
        if (comfy_root / "comfy" / "sd.py").exists():
            print(f"Skipping incomplete ComfyUI reference root: {comfy_root}")

    comfy_root = DEFAULT_COMFYUI_ROOT
    comfy_root.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            COMFYUI_REPO_URL,
            str(comfy_root),
        ],
        cwd=comfy_root.parent,
    )

    if not _comfy_root_has_required_modules(comfy_root):
        raise RuntimeError(
            "Cloned /content/ComfyUI but required comfy_aimdo modules were still missing. "
            "Use the bundled comfyui_tmp reference checkout from this branch."
        )

    os.environ["FOOOCUS_ANIMA_COMFY_ROOT"] = str(comfy_root)
    os.environ["ANIMA_COMFY_ROOT"] = str(comfy_root)
    print(f"FOOOCUS_ANIMA_COMFY_ROOT={comfy_root}")
    return comfy_root


def build_cases(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.profile == "baseline":
        return list(DEFAULT_CASES)
    if args.profile == "diagnostic":
        return list(DIAGNOSTIC_CASES)
    if args.profile == "official":
        return list(OFFICIAL_CASES)
    if args.profile == "worker_pair":
        return list(WORKER_PAIR_CASES)

    return [
        {
            "runner": "worker" if args.profile == "worker_single" else "plain",
            "label": args.label,
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "negative_mode": args.negative_mode,
            "steps": args.steps,
            "width": args.width,
            "height": args.height,
            "cfg": args.cfg,
            "seed": args.seed,
            "sampler": args.sampler,
            "scheduler": args.scheduler,
        }
    ]


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


def run_plain_case(case: dict[str, object], output_root: Path) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{case['label']}.png"

    cmd = [
        sys.executable,
        str(repo_file("tests", "test_anima_pipeline.py")),
        "--prompt",
        str(case["prompt"]),
        "--steps",
        str(case["steps"]),
        "--width",
        str(case["width"]),
        "--height",
        str(case["height"]),
        "--cfg",
        str(case["cfg"]),
        "--seed",
        str(case["seed"]),
        "--sampler",
        str(case.get("sampler", "euler")),
        "--scheduler",
        str(case.get("scheduler", "simple")),
        "--output",
        str(output_path),
    ]
    if case.get("negative_mode"):
        cmd.extend(["--negative-mode", str(case["negative_mode"])])
    if "negative_prompt" in case:
        cmd.extend(["--negative-prompt", str(case.get("negative_prompt", ""))])

    started = time.time()
    try:
        completed = run_capture(cmd, cwd=REPO_ROOT)
        metrics = compute_image_metrics(output_path)
        result = dict(case)
        result.update(
            {
                "status": "ok",
                "output_path": str(output_path),
                "duration_sec": round(time.time() - started, 2),
                "hf_metric": metrics["hf_metric"],
                "color_std": metrics["color_std"],
                "stdout_tail": completed.stdout.splitlines()[-20:],
            }
        )
        print("RESULT", json.dumps(result, ensure_ascii=False))
        return result
    except subprocess.CalledProcessError as exc:
        stdout = exc.stdout or ""
        result = dict(case)
        result.update(
            {
                "status": "error",
                "output_path": str(output_path),
                "duration_sec": round(time.time() - started, 2),
                "returncode": exc.returncode,
                "error": stdout.splitlines()[-20:],
            }
        )
        print("RESULT", json.dumps(result, ensure_ascii=False))
        return result


def run_worker_case(case: dict[str, object], output_root: Path) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{case['label']}.png"

    cmd = [
        sys.executable,
        str(WORKER_REPRO_FILE),
        "--label",
        str(case["label"]),
        "--prompt",
        str(case["prompt"]),
        "--negative-prompt",
        str(case.get("negative_prompt", "")),
        "--steps",
        str(case["steps"]),
        "--width",
        str(case["width"]),
        "--height",
        str(case["height"]),
        "--cfg",
        str(case["cfg"]),
        "--seed",
        str(case["seed"]),
        "--sampler",
        str(case.get("sampler", "euler")),
        "--scheduler",
        str(case.get("scheduler", "simple")),
        "--output",
        str(output_path),
        "--output-root",
        str(output_root),
    ]

    started = time.time()
    try:
        completed = run_capture(cmd, cwd=REPO_ROOT)
        metrics = compute_image_metrics(output_path)
        result = dict(case)
        result.update(
            {
                "status": "ok",
                "output_path": str(output_path),
                "duration_sec": round(time.time() - started, 2),
                "hf_metric": metrics["hf_metric"],
                "color_std": metrics["color_std"],
                "stdout_tail": completed.stdout.splitlines()[-20:],
            }
        )
        print("RESULT", json.dumps(result, ensure_ascii=False))
        return result
    except subprocess.CalledProcessError as exc:
        stdout = exc.stdout or ""
        result = dict(case)
        result.update(
            {
                "status": "error",
                "output_path": str(output_path),
                "duration_sec": round(time.time() - started, 2),
                "returncode": exc.returncode,
                "error": stdout.splitlines()[-20:],
            }
        )
        print("RESULT", json.dumps(result, ensure_ascii=False))
        return result


def run_case(case: dict[str, object], output_root: Path) -> dict[str, object]:
    if case.get("runner") == "worker":
        return run_worker_case(case, output_root)
    return run_plain_case(case, output_root)


def write_summary(output_root: Path, results: list[dict[str, object]]) -> Path:
    summary_path = output_root / "compare_results.json"
    payload = {
        "repo_root": str(REPO_ROOT),
        "output_root": str(output_root),
        "results": results,
    }
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fresh Colab bootstrap for Anima Preview2 headless reproduction."
    )
    parser.add_argument(
        "--profile",
        choices=["baseline", "diagnostic", "official", "single", "worker_pair", "worker_single"],
        default="baseline",
        help="Which case set to run after preparing the runtime.",
    )
    parser.add_argument("--label", default="plain_custom", help="Case label for --profile single.")
    parser.add_argument("--prompt", default="1girl", help="Prompt for --profile single.")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt for single profiles.")
    parser.add_argument(
        "--negative-mode",
        choices=["zero", "encode"],
        default="zero",
        help="Negative conditioning mode for plain single profiles.",
    )
    parser.add_argument("--steps", type=int, default=20, help="Steps for --profile single.")
    parser.add_argument("--width", type=int, default=1024, help="Width for --profile single.")
    parser.add_argument("--height", type=int, default=1024, help="Height for --profile single.")
    parser.add_argument("--cfg", type=float, default=4.0, help="CFG for --profile single.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for --profile single.")
    parser.add_argument("--sampler", default="euler", help="Sampler for single profiles.")
    parser.add_argument("--scheduler", default="simple", help="Scheduler for single profiles.")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory for generated images and compare_results.json.",
    )
    parser.add_argument(
        "--skip-requirements",
        action="store_true",
        help="Do not run pip install -r requirements_versions.txt.",
    )
    parser.add_argument(
        "--skip-downloads",
        action="store_true",
        help="Do not download Anima model files.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare the runtime. Do not run image generation cases.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_repo_root()
    ensure_required_repo_files()
    show_runtime_info()
    ensure_python_requirements(args.skip_requirements)
    ensure_models(args.skip_downloads)
    ensure_comfyui_reference()

    if args.prepare_only:
        print("Preparation finished. Cases were not executed.")
        return 0

    output_root = Path(args.output_root)
    results = [run_case(case, output_root) for case in build_cases(args)]
    summary_path = write_summary(output_root, results)

    print("\nSummary file:", summary_path)
    print("Generated files:")
    for result in results:
        print(" ", result.get("status"), result.get("output_path"))

    failures = [result for result in results if result.get("status") != "ok"]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
