#!/usr/bin/env python3
"""
Anima Preview2 パイプライン E2E テストスクリプト

Fooocus の Gradio UI を経由せず、パイプラインを直接実行して
テキストエンコード → 拡散サンプリング → VAE デコードの各段階を検証する。

使い方:
    # Colab (GPU) 上で実行
    cd /content/Fooocus
    python tests/test_anima_pipeline.py

    # オプション引数
    python tests/test_anima_pipeline.py \
        --prompt "1girl, anime style, blue eyes" \
        --steps 20 \
        --width 1024 --height 1024 \
        --output /content/test_output.png

必要ファイル:
    models/checkpoints/anima-preview2.safetensors  (4.18 GB)
    models/clip/qwen_3_06b_base.safetensors        (1.19 GB)
    models/vae/qwen_image_vae.safetensors           (0.25 GB)

注意:
    - T4 GPU (15GB VRAM) で動作確認済み
    - 拡散モデルをCPUオフロード後にVAEデコードする（VRAM節約）
    - VAEデコードはタイルドデコードに自動フォールバック
"""

import argparse
import os
import sys
import time

import numpy as np
import torch


def setup_paths():
    """Fooocus ルートをパスに追加"""
    fooocus_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if fooocus_root not in sys.path:
        sys.path.insert(0, fooocus_root)
    os.chdir(fooocus_root)
    # argparse 競合回避 (Jupyter/Colab 対応)
    original_argv = sys.argv
    sys.argv = ['fooocus']
    return fooocus_root, original_argv


def find_model_files(fooocus_root):
    """モデルファイルを探す"""
    paths = {
        'checkpoint': os.path.join(fooocus_root, 'models', 'checkpoints', 'anima-preview2.safetensors'),
        'clip': os.path.join(fooocus_root, 'models', 'clip', 'qwen_3_06b_base.safetensors'),
        'vae': os.path.join(fooocus_root, 'models', 'vae', 'qwen_image_vae.safetensors'),
    }
    print("=== モデルファイル確認 ===")
    all_found = True
    for name, path in paths.items():
        exists = os.path.exists(path)
        size = os.path.getsize(path) / 1024**3 if exists else 0
        status = f"✓ ({size:.2f} GB)" if exists else "✗ 見つかりません"
        print(f"  {name}: {status}")
        print(f"    {path}")
        if not exists:
            all_found = False
    return paths, all_found


def test_text_encoder(clip_path, prompt, max_length=256):
    """Step 1: テキストエンコーダーのテスト (Qwen3 + T5 デュアルトークナイザー)"""
    print("\n=== Step 1: テキストエンコーダー ===")

    from modules.anima_text_encoder import AnimaTextEncoder
    encoder = AnimaTextEncoder(clip_path)

    # 基本チェック
    assert encoder.model is not None, "Qwen3 モデルがロードされていません"
    assert encoder.qwen_tokenizer is not None, "Qwen3 トークナイザーがロードされていません"
    assert encoder.t5_tokenizer is not None, "T5 トークナイザーがロードされていません"
    print(f"  ✓ Qwen3 モデル + 両トークナイザー ロード完了")

    # エンコード
    t0 = time.time()
    hidden_states, token_ids = encoder.encode(prompt, max_length=max_length)
    t1 = time.time()

    print(f"  プロンプト: '{prompt}'")
    print(f"  hidden_states: shape={hidden_states.shape}, dtype={hidden_states.dtype}")
    print(f"    mean={hidden_states.float().mean():.4f}, std={hidden_states.float().std():.4f}")
    print(f"  token_ids: shape={token_ids.shape}, dtype={token_ids.dtype}")
    print(f"    max={token_ids.max().item()}, 非ゼロ数={token_ids.nonzero().shape[0]}")
    print(f"  エンコード時間: {t1-t0:.2f}秒")

    # 品質チェック
    assert hidden_states.shape == (1, max_length, 1024), \
        f"hidden_states の shape が不正: {hidden_states.shape}"
    assert token_ids.max().item() < 32128, \
        f"token_ids が T5 語彙範囲 (32128) を超えています: {token_ids.max().item()}"
    assert hidden_states.float().std() > 0.01, \
        "hidden_states がほぼゼロです（モデルが正しく動作していない可能性）"

    print(f"  ✓ テキストエンコーダー テスト合格")
    return encoder, hidden_states, token_ids


def test_model_loading(ckpt_path):
    """Step 2: Anima モデルのロード"""
    print("\n=== Step 2: モデルロード ===")

    import ldm_patched.modules.sd as sd_module
    from ldm_patched.modules.utils import load_torch_file

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
        model_patcher, clip, vae, vae_filename, clipvision = result
    elif len(result) == 4:
        model_patcher, clip, vae, clipvision = result
        vae_filename = None
    else:
        raise AssertionError(f"unexpected checkpoint result tuple length: {len(result)}")
    model = model_patcher.model

    print(f"  モデルタイプ: {type(model).__name__}")
    print(f"  拡散モデル: {type(model.diffusion_model).__name__}")
    print(f"  VAE: {vae is not None}")
    if vae_filename is not None:
        print(f"  外部VAE: {vae_filename}")
    print(f"  ロード時間: {t1-t0:.2f}秒")

    # VAE が自動ロードされたか確認
    assert vae is not None, \
        "VAE が None です。_find_anima_vae() が正しく動作していない可能性があります"
    print(f"  ✓ モデル + VAE ロード完了")

    return model_patcher, vae


def build_conditioning(hidden_states, token_ids, *, dtype=torch.float16):
    pooled = torch.zeros(1, 1024)
    cond = hidden_states.to(dtype=dtype)
    t5_ids_tensor = token_ids.long()
    if t5_ids_tensor.dim() == 2:
        t5_ids_tensor = t5_ids_tensor[0]
    t5_weights = torch.ones_like(t5_ids_tensor, dtype=torch.float32)
    return [cond, {
        "pooled_output": pooled,
        "t5xxl_ids": t5_ids_tensor,
        "t5xxl_weights": t5_weights,
    }]


def test_sampling(model_patcher, hidden_states, token_ids, steps=20,
                  width=1024, height=1024, cfg=4.0, seed=42,
                  negative_hidden_states=None, negative_token_ids=None):
    """Step 3: 拡散サンプリング"""
    print(f"\n=== Step 3: 拡散サンプリング ({steps} steps) ===")

    import ldm_patched.modules.sample as sample_module

    model = model_patcher.model
    dm = model.diffusion_model

    # ノイズ生成 (4D: batch, channels, height/8, width/8)
    latent_h, latent_w = height // 8, width // 8
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(1, 16, latent_h, latent_w, generator=generator)
    print(f"  ノイズ: shape={noise.shape}, seed={seed}")

    # コンディショニング準備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16

    positive = [build_conditioning(hidden_states, token_ids, dtype=dtype)]
    if negative_hidden_states is None or negative_token_ids is None:
        positive_cond, positive_meta = positive[0]
        negative = [[torch.zeros_like(positive_cond), {
            "pooled_output": torch.zeros_like(positive_meta["pooled_output"]),
            "t5xxl_ids": positive_meta["t5xxl_ids"].clone(),
            "t5xxl_weights": torch.zeros_like(positive_meta["t5xxl_weights"]),
        }]]
        print("  negative path: zero-filled conditioning")
    else:
        negative = [build_conditioning(
            negative_hidden_states,
            negative_token_ids,
            dtype=dtype,
        )]
        print("  negative path: encoded prompt conditioning")

    # サンプラー設定 (euler + simple, shift=3.0)
    print(f"  サンプラー: euler, スケジューラ: simple, CFG: {cfg}")

    # サンプリング実行
    t0 = time.time()
    samples = sample_module.sample(
        model=model_patcher,
        noise=noise,
        steps=steps,
        cfg=cfg,
        sampler_name="euler",
        scheduler="simple",
        positive=positive,
        negative=negative,
        latent_image=torch.zeros_like(noise),
        denoise=1.0,
        force_full_denoise=True,
        seed=seed,
    )
    t1 = time.time()

    # CPU にオフロード (VAE デコード用に VRAM 確保)
    dm.to('cpu')
    torch.cuda.empty_cache()

    print(f"  結果: shape={samples.shape}")
    print(f"    mean={samples.mean():.4f}, std={samples.std():.4f}")
    print(f"    min={samples.min():.4f}, max={samples.max():.4f}")
    print(f"  サンプリング時間: {t1-t0:.2f}秒")
    print(f"  GPU メモリ (オフロード後): {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # 品質チェック
    assert samples.std() > 0.01, "サンプル結果がほぼゼロです"
    assert not torch.isnan(samples).any(), "サンプル結果に NaN が含まれています"

    print(f"  ✓ サンプリング テスト合格")
    return samples


def test_vae_decode(vae, samples, output_path):
    """Step 4: VAE デコード + 画像保存"""
    print(f"\n=== Step 4: VAE デコード ===")

    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  GPU メモリ (デコード前): {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    t0 = time.time()
    decoded = vae.decode(samples)
    t1 = time.time()

    print(f"  デコード結果: shape={decoded.shape}, dtype={decoded.dtype}")
    print(f"    min={decoded.min():.3f}, max={decoded.max():.3f}, mean={decoded.mean():.3f}")
    print(f"  デコード時間: {t1-t0:.2f}秒")

    # 画像保存
    from PIL import Image
    img_array = (decoded[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    img.save(output_path)
    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"  保存先: {output_path} ({file_size:.1f} MB)")

    # 品質チェック
    pixels = img_array.reshape(-1, 3).astype(float)
    r_std, g_std, b_std = pixels[:, 0].std(), pixels[:, 1].std(), pixels[:, 2].std()
    print(f"  カラー標準偏差: R={r_std:.1f}, G={g_std:.1f}, B={b_std:.1f}")

    unique_colors = len(set(map(tuple, img_array.reshape(-1, 3)[:10000])))
    print(f"  ユニークカラー (10k サンプル): {unique_colors}")

    assert r_std > 10 and g_std > 10 and b_std > 10, \
        "カラー分布が単調です（ほぼ単色画像）"
    assert unique_colors > 100, \
        "ユニークカラーが少なすぎます"

    print(f"  ✓ VAE デコード テスト合格")
    print(f"\n  画像サイズ: {img.size}")
    return img


def main():
    parser = argparse.ArgumentParser(description="Anima Preview2 パイプライン E2E テスト")
    parser.add_argument("--prompt", default="1girl, solo, anime style, detailed face, long black hair, blue eyes, school uniform, cherry blossom",
                        help="生成プロンプト")
    parser.add_argument("--steps", type=int, default=20, help="サンプリングステップ数")
    parser.add_argument("--width", type=int, default=1024, help="画像幅")
    parser.add_argument("--height", type=int, default=1024, help="画像高さ")
    parser.add_argument("--cfg", type=float, default=4.0, help="CFG スケール")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")
    parser.add_argument("--output", default=None, help="出力画像パス")
    parser.add_argument("--negative-prompt", default="",
                        help="negative prompt used when --negative-mode encode")
    parser.add_argument("--negative-mode", choices=["zero", "encode"], default="zero",
                        help="how to build the negative conditioning path")
    args = parser.parse_args()

    fooocus_root, original_argv = setup_paths()

    if args.output is None:
        args.output = os.path.join(fooocus_root, 'tests', 'anima_test_output.png')

    print("=" * 60)
    print("Anima Preview2 パイプライン E2E テスト")
    print("=" * 60)

    # モデルファイル確認
    paths, all_found = find_model_files(fooocus_root)
    if not all_found:
        print("\n✗ モデルファイルが不足しています。テストを中断します。")
        sys.exit(1)

    # GPU 情報
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"\nGPU: {props.name}, VRAM: {props.total_memory / 1024**3:.1f} GB")
    else:
        print("\n⚠ GPU が利用できません。CPU で実行します（非常に遅い）。")

    total_t0 = time.time()

    # Step 1: テキストエンコーダー
    encoder, hidden_states, token_ids = test_text_encoder(
        paths['clip'], args.prompt, max_length=256
    )
    negative_hidden_states = None
    negative_token_ids = None
    if args.negative_mode == "encode":
        print("\n=== Step 1b: negative text encoder ===")
        negative_hidden_states, negative_token_ids = encoder.encode(
            args.negative_prompt, max_length=256
        )
        print(f"  negative prompt: '{args.negative_prompt}'")
        print(
            f"  negative hidden_states std={negative_hidden_states.float().std():.4f}, "
            f"token_count={negative_token_ids.nonzero().shape[0]}"
        )

    # Qwen3 モデルをオフロード（VRAM 節約）
    if encoder.model is not None:
        encoder.model.to('cpu')
    del encoder
    import gc; gc.collect()
    torch.cuda.empty_cache()

    # Step 2: モデルロード
    model_patcher, vae = test_model_loading(paths['checkpoint'])

    # Step 3: サンプリング
    samples = test_sampling(
        model_patcher, hidden_states, token_ids,
        steps=args.steps, width=args.width, height=args.height,
        cfg=args.cfg, seed=args.seed,
        negative_hidden_states=negative_hidden_states,
        negative_token_ids=negative_token_ids,
    )

    # model_patcher の拡散モデルはすでに CPU (test_sampling 内でオフロード済み)
    del model_patcher
    gc.collect()
    torch.cuda.empty_cache()

    # Step 4: VAE デコード
    img = test_vae_decode(vae, samples, args.output)

    total_t1 = time.time()

    print("\n" + "=" * 60)
    print(f"✓ 全テスト合格！ 合計時間: {total_t1-total_t0:.1f}秒")
    print(f"  出力画像: {args.output}")
    print("=" * 60)

    sys.argv = original_argv


if __name__ == "__main__":
    main()
