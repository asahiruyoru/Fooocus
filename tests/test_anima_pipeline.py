#!/usr/bin/env python3
"""
End-to-end Anima Preview2 sampling test.

This script avoids the Gradio UI and directly exercises:
1. text encoding
2. model and VAE loading
3. diffusion sampling
4. VAE decode and image save
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import numpy as np
import torch


def setup_paths() -> tuple[str, list[str]]:
    fooocus_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if fooocus_root not in sys.path:
        sys.path.insert(0, fooocus_root)
    os.chdir(fooocus_root)
    original_argv = sys.argv
    sys.argv = ["fooocus"]
    return fooocus_root, original_argv


def find_model_files(fooocus_root: str) -> tuple[dict[str, str], bool]:
    paths = {
        "checkpoint": os.path.join(
            fooocus_root, "models", "checkpoints", "anima-preview2.safetensors"
        ),
        "clip": os.path.join(
            fooocus_root, "models", "clip", "qwen_3_06b_base.safetensors"
        ),
        "vae": os.path.join(
            fooocus_root, "models", "vae", "qwen_image_vae.safetensors"
        ),
    }

    print("=== Model Files ===")
    all_found = True
    for name, path in paths.items():
        exists = os.path.exists(path)
        size_gb = os.path.getsize(path) / 1024**3 if exists else 0.0
        status = f"ok ({size_gb:.2f} GB)" if exists else "missing"
        print(f"  {name}: {status}")
        print(f"    {path}")
        if not exists:
            all_found = False
    return paths, all_found


def test_text_encoder(clip_path: str, prompt: str, max_length: int = 256):
    print("\n=== Step 1: Text Encoder ===")

    from modules.anima_text_encoder import AnimaTextEncoder

    encoder = AnimaTextEncoder(clip_path)

    assert encoder.model is not None, "Qwen3 text encoder did not load"
    assert encoder.qwen_tokenizer is not None, "Qwen3 tokenizer did not load"
    assert encoder.t5_tokenizer is not None, "T5 tokenizer did not load"

    t0 = time.time()
    encoded = encoder.encode(prompt, max_length=max_length)
    t1 = time.time()

    if len(encoded) == 2:
        hidden_states, token_ids = encoded
        token_weights = None
    else:
        hidden_states, token_ids, token_weights = encoded

    print(f"  prompt: {prompt!r}")
    print(f"  hidden_states: shape={tuple(hidden_states.shape)}, dtype={hidden_states.dtype}")
    print(
        f"    mean={hidden_states.float().mean():.4f}, "
        f"std={hidden_states.float().std():.4f}"
    )
    print(f"  token_ids: shape={tuple(token_ids.shape)}, dtype={token_ids.dtype}")
    print(f"    max={token_ids.max().item()}, nonzero={token_ids.nonzero().shape[0]}")
    if token_weights is not None:
        print(
            f"  token_weights: shape={tuple(token_weights.shape)}, "
            f"dtype={token_weights.dtype}"
        )
    print(f"  encode_time_sec={t1 - t0:.2f}")

    assert hidden_states.shape[0] == 1 and hidden_states.shape[-1] == 1024, (
        f"unexpected hidden state shape: {hidden_states.shape}"
    )
    assert token_ids.max().item() < 32128, (
        f"token ids exceed T5 vocab limit: {token_ids.max().item()}"
    )
    if token_weights is not None:
        assert token_ids.numel() == token_weights.numel(), (
            "token_ids and token_weights length mismatch"
        )
    assert hidden_states.float().std() > 0.01, "hidden states are nearly zero"

    print("  text encoder test passed")
    return encoder, hidden_states, token_ids, token_weights


def test_model_loading(ckpt_path: str):
    print("\n=== Step 2: Model Load ===")

    import ldm_patched.modules.sd as sd_module

    t0 = time.time()
    result = sd_module.load_checkpoint_guess_config(
        ckpt_path,
        output_vae=True,
        output_clip=False,
        output_clipvision=False,
        embedding_directory=None,
    )
    t1 = time.time()

    if len(result) == 5:
        model_patcher, _clip, vae, vae_filename, _clipvision = result
    elif len(result) == 4:
        model_patcher, _clip, vae, _clipvision = result
        vae_filename = None
    else:
        raise AssertionError(f"unexpected checkpoint result tuple length: {len(result)}")

    model = model_patcher.model
    print(f"  model_type={type(model).__name__}")
    print(f"  diffusion_model={type(model.diffusion_model).__name__}")
    print(f"  vae_loaded={vae is not None}")
    if vae_filename is not None:
        print(f"  separate_vae={vae_filename}")
    print(f"  load_time_sec={t1 - t0:.2f}")

    assert vae is not None, "VAE did not load"
    print("  model load test passed")
    return model_patcher, vae


def build_conditioning(
    hidden_states: torch.Tensor,
    token_ids: torch.Tensor,
    token_weights: torch.Tensor | None = None,
    *,
    dtype: torch.dtype = torch.float16,
):
    pooled = torch.zeros(1, 1024)
    cond = hidden_states.to(dtype=dtype)
    t5_ids_tensor = token_ids.long()
    if t5_ids_tensor.dim() == 2:
        t5_ids_tensor = t5_ids_tensor[0]

    if token_weights is None:
        t5_weights = torch.ones_like(t5_ids_tensor, dtype=torch.float32)
    else:
        t5_weights = token_weights.to(dtype=torch.float32)
        if t5_weights.dim() == 2:
            t5_weights = t5_weights[0]

    return [
        cond,
        {
            "pooled_output": pooled,
            "t5xxl_ids": t5_ids_tensor,
            "t5xxl_weights": t5_weights,
        },
    ]


def test_sampling(
    model_patcher,
    hidden_states: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    steps: int = 20,
    width: int = 1024,
    height: int = 1024,
    cfg: float = 4.0,
    seed: int = 42,
    token_weights: torch.Tensor | None = None,
    negative_hidden_states: torch.Tensor | None = None,
    negative_token_ids: torch.Tensor | None = None,
    negative_token_weights: torch.Tensor | None = None,
    sampler_name: str = "euler",
    scheduler_name: str = "simple",
):
    print(f"\n=== Step 3: Sampling ({steps} steps) ===")

    import ldm_patched.modules.sample as sample_module

    latent_h, latent_w = height // 8, width // 8
    latent_format = model_patcher.model.latent_format
    latent_channels = getattr(latent_format, "latent_channels", 16)
    latent_dimensions = getattr(latent_format, "latent_dimensions", 2)
    latent_shape = [1, latent_channels, latent_h, latent_w]
    if latent_dimensions == 3:
        latent_shape = [1, latent_channels, 1, latent_h, latent_w]

    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(*latent_shape, generator=generator)
    dtype = torch.float16

    positive = [build_conditioning(hidden_states, token_ids, token_weights, dtype=dtype)]
    if negative_hidden_states is None or negative_token_ids is None:
        positive_cond, positive_meta = positive[0]
        negative = [[
            torch.zeros_like(positive_cond),
            {
                "pooled_output": torch.zeros_like(positive_meta["pooled_output"]),
                "t5xxl_ids": positive_meta["t5xxl_ids"].clone(),
                "t5xxl_weights": torch.zeros_like(positive_meta["t5xxl_weights"]),
            },
        ]]
        print("  negative path: zero-filled conditioning")
    else:
        negative = [
            build_conditioning(
                negative_hidden_states,
                negative_token_ids,
                negative_token_weights,
                dtype=dtype,
            )
        ]
        print("  negative path: encoded prompt conditioning")

    print(
        f"  sampler={sampler_name}, scheduler={scheduler_name}, "
        f"cfg={cfg}, seed={seed}"
    )
    print(f"  noise_shape={tuple(noise.shape)}")

    t0 = time.time()
    samples = sample_module.sample(
        model=model_patcher,
        noise=noise,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler_name,
        scheduler=scheduler_name,
        positive=positive,
        negative=negative,
        latent_image=torch.zeros_like(noise),
        denoise=1.0,
        force_full_denoise=True,
        seed=seed,
    )
    t1 = time.time()

    model_patcher.model.diffusion_model.to("cpu")
    torch.cuda.empty_cache()

    print(f"  samples_shape={tuple(samples.shape)}")
    print(
        f"  stats: mean={samples.mean():.4f}, std={samples.std():.4f}, "
        f"min={samples.min():.4f}, max={samples.max():.4f}"
    )
    print(f"  sampling_time_sec={t1 - t0:.2f}")
    if torch.cuda.is_available():
        print(f"  gpu_mem_after_offload_gb={torch.cuda.memory_allocated() / 1024**3:.2f}")

    assert samples.std() > 0.01, "sample output is nearly zero"
    assert not torch.isnan(samples).any(), "sample output contains NaN"
    print("  sampling test passed")
    return samples


def test_vae_decode(vae, samples: torch.Tensor, output_path: str):
    print("\n=== Step 4: VAE Decode ===")

    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print(f"  gpu_mem_before_decode_gb={torch.cuda.memory_allocated() / 1024**3:.2f}")

    t0 = time.time()
    decoded = vae.decode(samples)
    t1 = time.time()

    print(f"  decoded_shape={tuple(decoded.shape)}, dtype={decoded.dtype}")
    print(
        f"  decoded_stats: min={decoded.min():.3f}, max={decoded.max():.3f}, "
        f"mean={decoded.mean():.3f}"
    )
    print(f"  decode_time_sec={t1 - t0:.2f}")

    from PIL import Image

    img_array = (decoded[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    return img


def save_test_image_with_metadata(
    img,
    output_path: str,
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
):
    from modules.a1111_png_metadata import save_png_with_a1111_metadata

    save_png_with_a1111_metadata(
        img,
        output_path,
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        width=width,
        height=height,
        cfg=cfg,
        seed=seed,
        sampler=sampler,
        scheduler=scheduler,
        base_model_name="anima-preview2.safetensors",
        vae_name="qwen_image_vae.safetensors",
        performance="Quality" if steps >= 40 else None,
        full_prompt=[prompt],
        full_negative_prompt=[negative_prompt] if negative_prompt else [],
    )

    img_array = np.array(img)
    file_size_mb = os.path.getsize(output_path) / 1024 / 1024
    pixels = img_array.reshape(-1, 3).astype(float)
    r_std, g_std, b_std = pixels[:, 0].std(), pixels[:, 1].std(), pixels[:, 2].std()
    unique_colors = len(set(map(tuple, img_array.reshape(-1, 3)[:10000])))

    print(f"  saved={output_path} ({file_size_mb:.1f} MB)")
    print(f"  channel_std: R={r_std:.1f}, G={g_std:.1f}, B={b_std:.1f}")
    print(f"  unique_colors_10k={unique_colors}")

    assert r_std > 10 and g_std > 10 and b_std > 10, "decoded image lacks color variation"
    assert unique_colors > 100, "decoded image has too few unique colors"

    print(f"  image_size={img.size}")
    print("  VAE decode test passed")
    return img


def main() -> int:
    parser = argparse.ArgumentParser(description="Anima Preview2 E2E sampling test")
    parser.add_argument(
        "--prompt",
        default=(
            "1girl, solo, anime style, detailed face, long black hair, blue eyes, "
            "school uniform, cherry blossom"
        ),
        help="Positive prompt",
    )
    parser.add_argument("--steps", type=int, default=20, help="Sampling steps")
    parser.add_argument("--width", type=int, default=1024, help="Output width")
    parser.add_argument("--height", type=int, default=1024, help="Output height")
    parser.add_argument("--cfg", type=float, default=4.0, help="CFG scale")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--output", default=None, help="Output image path")
    parser.add_argument(
        "--negative-prompt",
        default="",
        help="Negative prompt used when --negative-mode encode",
    )
    parser.add_argument(
        "--negative-mode",
        choices=["zero", "encode"],
        default="zero",
        help="How to build the negative conditioning path",
    )
    parser.add_argument("--sampler", default="euler", help="Sampler name")
    parser.add_argument("--scheduler", default="simple", help="Scheduler name")
    args = parser.parse_args()

    fooocus_root, original_argv = setup_paths()

    if args.output is None:
        args.output = os.path.join(fooocus_root, "tests", "anima_test_output.png")

    print("=" * 60)
    print("Anima Preview2 E2E sampling test")
    print("=" * 60)

    paths, all_found = find_model_files(fooocus_root)
    if not all_found:
        print("\nMissing required model files.")
        return 1

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"\nGPU: {props.name}, VRAM: {props.total_memory / 1024**3:.1f} GB")
    else:
        print("\nGPU is not available.")

    total_t0 = time.time()

    encoder, hidden_states, token_ids, token_weights = test_text_encoder(
        paths["clip"], args.prompt, max_length=256
    )

    negative_hidden_states = None
    negative_token_ids = None
    negative_token_weights = None
    if args.negative_mode == "encode":
        print("\n=== Step 1b: Negative Text Encoder ===")
        encoded_negative = encoder.encode(args.negative_prompt, max_length=256)
        if len(encoded_negative) == 2:
            negative_hidden_states, negative_token_ids = encoded_negative
            negative_token_weights = None
        else:
            (
                negative_hidden_states,
                negative_token_ids,
                negative_token_weights,
            ) = encoded_negative

        print(f"  negative_prompt: {args.negative_prompt!r}")
        print(
            f"  negative_hidden_std={negative_hidden_states.float().std():.4f}, "
            f"token_count={negative_token_ids.nonzero().shape[0]}"
        )

    if encoder.model is not None:
        encoder.model.to("cpu")
    del encoder
    gc.collect()
    torch.cuda.empty_cache()

    model_patcher, vae = test_model_loading(paths["checkpoint"])

    samples = test_sampling(
        model_patcher,
        hidden_states,
        token_ids,
        steps=args.steps,
        width=args.width,
        height=args.height,
        cfg=args.cfg,
        seed=args.seed,
        token_weights=token_weights,
        negative_hidden_states=negative_hidden_states,
        negative_token_ids=negative_token_ids,
        negative_token_weights=negative_token_weights,
        sampler_name=args.sampler,
        scheduler_name=args.scheduler,
    )

    del model_patcher
    gc.collect()
    torch.cuda.empty_cache()

    img = test_vae_decode(vae, samples, args.output)
    save_test_image_with_metadata(
        img,
        args.output,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        steps=args.steps,
        width=args.width,
        height=args.height,
        cfg=args.cfg,
        seed=args.seed,
        sampler=args.sampler,
        scheduler=args.scheduler,
    )

    total_t1 = time.time()
    print("\n" + "=" * 60)
    print(f"All steps passed in {total_t1 - total_t0:.1f}s")
    print(f"Output image: {args.output}")
    print("=" * 60)

    sys.argv = original_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
