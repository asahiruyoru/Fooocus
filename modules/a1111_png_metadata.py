from __future__ import annotations

from pathlib import Path
from typing import Sequence

import fooocus_version
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from modules.flags import CIVITAI_NO_KARRAS, MetadataScheme, Performance, SAMPLERS
from modules.util import quote


def _normalize_prompt_list(prompt: str | Sequence[str] | None) -> list[str]:
    if prompt is None:
        return []
    if isinstance(prompt, str):
        return [prompt] if prompt else []
    return [item for item in prompt if item]


def _joined_prompt(prompt: str | Sequence[str] | None, fallback: str = "") -> str:
    parts = _normalize_prompt_list(prompt)
    return ", ".join(parts) if parts else fallback


def _format_sampler(sampler: str, scheduler: str) -> str:
    sampler_text = SAMPLERS.get(sampler, sampler) or sampler
    if sampler not in CIVITAI_NO_KARRAS and scheduler == "karras":
        sampler_text += " Karras"
    return sampler_text


def build_a1111_parameters(
    *,
    prompt: str,
    negative_prompt: str,
    steps: int,
    width: int,
    height: int,
    cfg: float,
    seed: int,
    sampler: str,
    scheduler: str,
    base_model_name: str,
    vae_name: str,
    performance: str | None = None,
    sharpness: float = 2.0,
    prompt_expansion: bool = False,
    styles: Sequence[str] | None = None,
    full_prompt: str | Sequence[str] | None = None,
    full_negative_prompt: str | Sequence[str] | None = None,
    adm_guidance: tuple[float, float, float] = (1.5, 0.8, 0.3),
    refiner_model_name: str = "None",
    refiner_switch: float = 0.5,
    loras: Sequence[tuple[str, float]] | None = None,
) -> str:
    if performance is None:
        try:
            performance = Performance.by_steps(steps).value
        except ValueError:
            performance = None

    positive_text = _joined_prompt(full_prompt if full_prompt is not None else prompt, prompt)
    negative_text = _joined_prompt(
        full_negative_prompt if full_negative_prompt is not None else negative_prompt,
        negative_prompt,
    )

    params: list[tuple[str, object]] = [
        ("Steps", steps),
        ("Sampler", _format_sampler(sampler, scheduler)),
        ("Seed", seed),
        ("Size", f"{width}x{height}"),
        ("CFG scale", cfg),
        ("Sharpness", sharpness),
        ("ADM Guidance", str(adm_guidance)),
        ("Model", Path(base_model_name).stem),
    ]

    if performance is not None:
        params.append(("Performance", performance))

    params.append(("Scheduler", scheduler))
    params.append(("VAE", Path(vae_name).stem))

    style_values = list(styles or [])
    if style_values:
        params.append(("Styles", str(style_values)))

    if prompt_expansion:
        params.append(("Fooocus V2 Expansion", prompt_expansion))

    if refiner_model_name not in ("", "None"):
        params.append(("Refiner", Path(refiner_model_name).stem))
        params.append(("Refiner Switch", refiner_switch))

    for lora_name, lora_weight in loras or []:
        if lora_name != "None":
            params.append(("LoRA", f"{Path(lora_name).stem}: {lora_weight}"))

    params.append(("Version", "Fooocus v" + fooocus_version.version))

    params_text = ", ".join(f"{key}: {quote(value)}" for key, value in params if value is not None)
    negative_line = f"\nNegative prompt: {negative_text}" if negative_text else ""
    return f"{positive_text}{negative_line}\n{params_text}".strip()


def save_png_with_a1111_metadata(
    image: Image.Image,
    output_path: str | Path,
    *,
    prompt: str,
    negative_prompt: str,
    steps: int,
    width: int,
    height: int,
    cfg: float,
    seed: int,
    sampler: str,
    scheduler: str,
    base_model_name: str,
    vae_name: str,
    performance: str | None = None,
    sharpness: float = 2.0,
    prompt_expansion: bool = False,
    styles: Sequence[str] | None = None,
    full_prompt: str | Sequence[str] | None = None,
    full_negative_prompt: str | Sequence[str] | None = None,
    adm_guidance: tuple[float, float, float] = (1.5, 0.8, 0.3),
    refiner_model_name: str = "None",
    refiner_switch: float = 0.5,
    loras: Sequence[tuple[str, float]] | None = None,
) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parameters = build_a1111_parameters(
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        width=width,
        height=height,
        cfg=cfg,
        seed=seed,
        sampler=sampler,
        scheduler=scheduler,
        base_model_name=base_model_name,
        vae_name=vae_name,
        performance=performance,
        sharpness=sharpness,
        prompt_expansion=prompt_expansion,
        styles=styles,
        full_prompt=full_prompt,
        full_negative_prompt=full_negative_prompt,
        adm_guidance=adm_guidance,
        refiner_model_name=refiner_model_name,
        refiner_switch=refiner_switch,
        loras=loras,
    )

    pnginfo = PngInfo()
    pnginfo.add_text("parameters", parameters)
    pnginfo.add_text("fooocus_scheme", MetadataScheme.A1111.value)
    image.save(output_path, pnginfo=pnginfo)
    return parameters


def rewrite_png_with_a1111_metadata(
    image_path: str | Path,
    **kwargs,
) -> str:
    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")
    return save_png_with_a1111_metadata(image, image_path, **kwargs)
