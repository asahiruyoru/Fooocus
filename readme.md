# Fooocus (拡張フォーク版)

> mashb1t の 1-Up Edition をベースに、**Anima Preview2 対応**や UI 改善を加えたフォークです。

---

## 最近の変更点

### Anima Preview2 対応 (実験的)

CircleStone Labs の **Anima Preview2** モデルファミリーを Fooocus から使えるようにしました。

| 項目 | 内容 |
|------|------|
| バックエンド | DiT / Qwen3 テキストエンコーダー / Wan VAE のロードパスを追加 |
| サンプラー修正 | ワーカーパスを公式リファレンスサンプラーに合わせて修正 (euler_ancestral + simple) |
| プリセット | `anima_preview2` プリセット追加 (1344x1344 / 40 steps / CFG 4.5) |
| デフォルトプロンプト | Anima のタグ式プロンプトに合わせた初期プロンプトを設定 |
| 安全策 | Fooocus V2 プロンプト拡張の自動無効化、未対応サンプラー/スケジューラの自動フォールバック |
| メタデータ | A1111 形式のメタデータを再現用画像に埋め込み |
| Colab 対応 | `scripts/anima_preview2_colab_bootstrap.py` でモデルDL〜診断まで一括実行 |
| Balanced モード | 40 ステップのバランスドパフォーマンス設定を追加 |
| 起動修正 | `rembg` を遅延ロードにして、オプション依存がなくても起動可能に |

### ControlNet: AnyTest 対応

- ControlNet モデルを **PyraCanny/CPDS → AnyTest/AnyTest_B** に置き換え
- Classic と AnyTest の ControlNet を**同時利用可能**に
- ControlNet スロットを全タイプ分に増設
- WD14 Tagger を **v3 (wd-eva02-large-tagger-v3)** にアップグレード

### その他の UI 改善

- アスペクト比パネルにカスタム幅/高さ入力を追加
- 高解像度アスペクト比プリセット追加 (1344, 1360, 1536, 1664)

### Notebook / Colab

- Animayume ノートブックプロファイルの追加と各種プリセット整備
- `--always-gpu` を T4/L4 クラス GPU でのデフォルト推奨に変更

---

## Fooocus とは

<div align=center>
<img src="https://github.com/lllyasviel/Fooocus/assets/19834515/483fb86d-c9a2-4c20-997c-46dafc124f25">
</div>

**Fooocus** は、プロンプトを入力するだけで高品質な画像を生成できるオープンソースソフトウェアです。

- Midjourney のように「プロンプトだけに集中」できる設計
- ダウンロードから初回生成まで **マウスクリック 3 回以内**
- 最低 GPU メモリ **4GB** (Nvidia) で動作
- オフライン・無料・オープンソース

> **注意**: このリポジトリは [lllyasviel/Fooocus](https://github.com/lllyasviel/Fooocus) → [mashb1t/Fooocus](https://github.com/mashb1t/Fooocus) からの**フォーク**です。公式リポジトリではありません。

### プロジェクトの状態

- **SDXL ワークフロー**: バグ修正のみの長期サポート (LTS) 状態
- **Anima Preview2**: 実験的サポート (上級者向け、SDXL ワークフローの完全な代替ではありません)
- **Flux 等の新モデル**: [WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) や [ComfyUI](https://github.com/comfyanonymous/ComfyUI) を推奨

---

## インストール

### Windows

Windows 向けビルド済みパッケージは [mashb1t/Fooocus のリリースページ](https://github.com/mashb1t/Fooocus/releases) から入手できます (本フォークのビルドではありません)。

このフォークを使う場合は git clone してください:
```bash
git clone https://github.com/mashb1t/Fooocus.git
cd Fooocus
```

初回起動時にモデルが自動ダウンロードされます。

プリセット別の起動バッチ:
- `run.bat` — 汎用 (Juggernaut XL v8)
- `run_anime.bat` — アニメ (animaPencilXL v500)
- `run_realistic.bat` — リアル系 (realisticStockPhoto v20)

> **"MetadataIncompleteBuffer" や "PytorchStreamReader" エラー**が出る場合はモデルファイルが壊れています。再ダウンロードしてください。

<details>
<summary>低スペックでも動く? (参考: 16GB RAM / 6GB VRAM / RTX 3060 Laptop)</summary>

約 1.35 秒/iteration で動作確認済み。

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/938737a5-b105-4f19-b051-81356cb7c495)

Nvidia ドライバ 532 以上で遅くなる場合は [Driver 531](https://www.nvidia.com/download/driverResults.aspx/199991/en-us/) を試してください。
</details>

<details>
<summary>仮想メモリ (スワップ) の設定</summary>

"RuntimeError: CPUAllocator" が出る場合は仮想メモリを有効にしてください。各ドライブに 40GB 以上の空き容量が必要です。

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/2a06b130-fe9b-4504-94f1-2763be4476e9)
</details>

### mashb1t フォークからの移行

```bash
# Fooocus フォルダでターミナルを開く
git remote set-url origin https://github.com/mashb1t/Fooocus.git
git reset --hard origin/main
git pull
# Python パッケージを更新
..\python_embeded\python.exe -m pip install -r "requirements_versions.txt"
```

### Colab (SDXL)

SDXL 版の Colab ノートブックは元の [mashb1t/Fooocus](https://github.com/mashb1t/Fooocus) で提供されています。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mashb1t/Fooocus/blob/main/fooocus_colab.ipynb) (mashb1t 版)

```bash
# 起動例
!python entry_with_update.py --share --always-high-vram --preset anime
```

### Colab (Anima Preview2)

```bash
git clone --depth 1 \
  https://github.com/asahiruyoru/Fooocus /content/Fooocus
cd /content/Fooocus
python scripts/anima_preview2_colab_bootstrap.py
python entry_with_update.py --share --always-gpu --preset anima_preview2
```

bootstrap スクリプトが行うこと:
- Python 依存パッケージのインストール
- Anima チェックポイント・VAE・テキストエンコーダーのダウンロード
- ヘッドレス診断の実行 (`/content/anima_case_outputs` に出力)

> T4/L4 クラス GPU では `--always-gpu` 推奨 (ピーク VRAM 約 11.4GB)。VRAM が足りない場合は `--always-high-vram` にフォールバック。

### Linux

<details>
<summary>Anaconda</summary>

```bash
git clone https://github.com/lllyasviel/Fooocus.git
cd Fooocus
conda env create -f environment.yaml
conda activate fooocus
pip install -r requirements_versions.txt
python entry_with_update.py
```
</details>

<details>
<summary>Python venv (Python 3.10)</summary>

```bash
git clone https://github.com/lllyasviel/Fooocus.git
cd Fooocus
python3 -m venv fooocus_env
source fooocus_env/bin/activate
pip install -r requirements_versions.txt
python entry_with_update.py
```
</details>

<details>
<summary>AMD GPU (Linux: ROCm / Windows: DirectML)</summary>

```bash
pip uninstall torch torchvision torchaudio torchtext functorch xformers
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
```

Windows の場合は `run.bat` を以下に書き換え:
```bat
.\python_embeded\python.exe -m pip uninstall torch torchvision torchaudio torchtext functorch xformers -y
.\python_embeded\python.exe -m pip install torch-directml
.\python_embeded\python.exe -s Fooocus\entry_with_update.py --directml
pause
```

AMD サポートはベータです。
</details>

<details>
<summary>Mac (Apple Silicon M1/M2)</summary>

1. conda と PyTorch nightly をインストール ([Apple ガイド](https://developer.apple.com/metal/pytorch/))
2. `git clone https://github.com/lllyasviel/Fooocus.git && cd Fooocus`
3. `conda env create -f environment.yaml && conda activate fooocus`
4. `pip install -r requirements_versions.txt`
5. `python entry_with_update.py`

M2 で遅い場合は `--disable-offload-from-vram` を追加。Nvidia RTX 3060 比で約 9 倍遅い。
</details>

<details>
<summary>Docker</summary>

[docker.md](docker.md) を参照。
</details>

---

## 最低動作要件

| OS | GPU | 最低 VRAM | 最低 RAM | スワップ | 備考 |
|----|-----|-----------|----------|----------|------|
| Win/Linux | Nvidia RTX 4XXX | 4GB | 8GB | 必須 | 最速 |
| Win/Linux | Nvidia RTX 3XXX | 4GB | 8GB | 必須 | |
| Win/Linux | Nvidia RTX 2XXX | 4GB | 8GB | 必須 | |
| Win/Linux | Nvidia GTX 1XXX | 8GB (6GB は不確実) | 8GB | 必須 | CPU とほぼ同速 |
| Windows | AMD GPU | 8GB | 8GB | 必須 | DirectML、RTX 3XXX の約 3 倍遅い |
| Linux | AMD GPU | 8GB | 8GB | 必須 | ROCm、RTX 3XXX の約 1.5 倍遅い |
| Mac | M1/M2 | 共有 | 共有 | 共有 | RTX 3XXX の約 9 倍遅い |
| 全 OS | CPU のみ | 0GB | 32GB | 必須 | RTX 3XXX の約 17 倍遅い |

トラブルシューティング: [troubleshoot.md](troubleshoot.md)

---

## 主な機能

### Midjourney との対応表

| Midjourney | Fooocus |
|------------|---------|
| テキストから高品質画像生成 | GPT-2 プロンプト拡張 + サンプリング最適化で、短いプロンプトでも高品質 |
| V1〜V4 (バリエーション) | Input Image → Upscale or Variation → Vary (Subtle/Strong) |
| U1〜U4 (アップスケール) | Input Image → Upscale or Variation → Upscale (1.5x/2x) |
| Inpaint / Pan | Input Image → Inpaint or Outpaint (独自アルゴリズム) |
| Image Prompt | Input Image → Image Prompt (独自アルゴリズム) |
| --style | Advanced → Style |
| --stylize | Advanced → Advanced → Guidance |
| --niji | `run_anime.bat` / `--preset anime` |
| --quality | Advanced → Quality |
| --repeat | Advanced → Image Number |
| Multi Prompts (::) | 複数行プロンプト |
| Prompt Weights | `(happy:1.5)` 形式 (A1111 互換) |
| --no | Advanced → Negative Prompt |
| --ar | Advanced → Aspect Ratios |
| InsightFace | Input Image → Image Prompt → Advanced → FaceSwap |
| Describe | Input Image → Describe |

### 自動マスク生成 + インペイント

rembg によるマスク自動生成、衣服カテゴリ別/プロンプト別のセグメンテーション対応。

### メタデータ (A1111/Civitai 互換)

- PNG (PngInfo) / JPG・WebP (EXIF) にメタデータ埋め込み
- Fooocus 形式 (JSON) と A1111 形式 (プレーンテキスト) をサポート
- 画像からパラメータを読み込んで再利用可能
- `--disable-metadata` で完全無効化

<details>
<summary>設定例 (config.txt)</summary>

```json
"default_save_metadata_to_images": true,
"default_metadata_scheme": "a1111",
"metadata_created_by": "your_name"
```
</details>

### Enhance (自動アップスケール + 補正)

adetailer に似た機能。動的画像検出ベースでワンクリック。[詳細](https://github.com/mashb1t/Fooocus/discussions/42)

---

## Anima Preview2 の詳細

### やったこと

- DiT / Qwen3 / Wan VAE のロードパスを Fooocus のワーカーに統合
- `ldm_patched/` 配下に必要なサポートファイルを追加 (`common_dit.py`, `cosmos/*`, `patcher_extension.py`)
- Anima 用プリセットを追加 (推奨デフォルト値つき)
- ワーカーのサンプリングパスを修正し、リファレンスサンプラーと同等の挙動に
- UNet パッチャーのクローン時に `model_file` を保持
- Anima で安全でないデフォルト組み合わせを自動クランプ

### ComfyUI を組み込んだの?

**いいえ。** UI・タスクキュー・プリセットシステムはすべて従来の Fooocus のまま。

`ldm_patched/` 内部の Comfy 由来コードを、Anima が期待するリファレンスサンプラーの挙動に合わせて修正しただけです。Colab 診断時に ComfyUI をリファレンス実装として比較に使いましたが、本アプリは Fooocus です。

### 推奨デフォルト値

| 項目 | 値 |
|------|----|
| サンプラー | `euler_ancestral` |
| スケジューラ | `simple` |
| ステップ数 | `40` |
| CFG | `4.5` |
| 解像度 | `1344x1344` |
| Colab フラグ | `--always-gpu` |

<details>
<summary>デフォルトプロンプト</summary>

**Positive:**
```
masterpiece, best quality, highres, safe, 1girl, solo, looking at viewer, smile, long hair, detailed eyes, detailed face, clean lines, smooth shading, soft lighting
```

**Negative:**
```
worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, watermark, patreon logo, bad hands, bad fingers, bad eyes, bad pupils, bad iris, 6 fingers, 6 toes
```
</details>

---

## カスタマイズ

初回起動後に生成される `config.txt` を編集してモデルパスやデフォルトパラメータを変更できます。壊した場合は `config.txt` を削除すればデフォルトに戻ります。

<details>
<summary>config.txt の例</summary>

```json
{
    "path_checkpoints": "D:\\Fooocus\\models\\checkpoints",
    "path_loras": "D:\\Fooocus\\models\\loras",
    "path_embeddings": "D:\\Fooocus\\models\\embeddings",
    "path_outputs": "D:\\Fooocus\\outputs",
    "default_model": "realisticStockPhoto_v10.safetensors",
    "default_refiner": "",
    "default_loras": [["lora_filename.safetensors", 0.5]],
    "default_cfg_scale": 3.0,
    "default_sampler": "dpmpp_2m",
    "default_scheduler": "karras",
    "default_negative_prompt": "low quality",
    "default_positive_prompt": "",
    "default_styles": ["Fooocus V2", "Fooocus Photograph", "Fooocus Negative"]
}
```

他のキーは初回起動後に生成される `config_modification_tutorial.txt` を参照。
</details>

## UI アクセスと認証

| 方法 | フラグ | 備考 |
|------|--------|------|
| ローカル公開 | `--listen` (ポート指定: `--port 8888`) | LAN 内アクセス |
| 外部公開 | `--share` | `.gradio.live` エンドポイント |

認証を有効にするには `auth.json` を作成 ([auth-example.json](./auth-example.json) 参照)。

## プロンプトの便利機能

| 機能 | 書き方 | 説明 |
|------|--------|------|
| ワイルドカード | `__color__ flower` | `wildcards/color.txt` からランダム選択 (シードベース) |
| 配列処理 | `[[red, green, blue]] flower` | 要素ごとに 1 枚ずつ生成 |
| インライン LoRA | `flower <lora:sunflowers:1.2>` | `models/loras/` 内の LoRA を適用 |

## コマンドラインフラグ一覧

<details>
<summary>全フラグを表示</summary>

```
entry_with_update.py  [-h] [--listen [IP]] [--port PORT]
                      [--disable-header-check [ORIGIN]]
                      [--web-upload-size WEB_UPLOAD_SIZE]
                      [--hf-mirror HF_MIRROR]
                      [--external-working-path PATH [PATH ...]]
                      [--output-path OUTPUT_PATH]
                      [--temp-path TEMP_PATH] [--cache-path CACHE_PATH]
                      [--in-browser] [--disable-in-browser]
                      [--gpu-device-id DEVICE_ID]
                      [--async-cuda-allocation | --disable-async-cuda-allocation]
                      [--disable-attention-upcast]
                      [--all-in-fp32 | --all-in-fp16]
                      [--unet-in-bf16 | --unet-in-fp16 | --unet-in-fp8-e4m3fn | --unet-in-fp8-e5m2]
                      [--vae-in-fp16 | --vae-in-fp32 | --vae-in-bf16]
                      [--vae-in-cpu]
                      [--clip-in-fp8-e4m3fn | --clip-in-fp8-e5m2 | --clip-in-fp16 | --clip-in-fp32]
                      [--directml [DIRECTML_DEVICE]]
                      [--disable-ipex-hijack]
                      [--preview-option [none,auto,fast,taesd]]
                      [--attention-split | --attention-quad | --attention-pytorch]
                      [--disable-xformers]
                      [--always-gpu | --always-high-vram | --always-normal-vram
                       | --always-low-vram | --always-no-vram | --always-cpu [CPU_NUM_THREADS]]
                      [--always-offload-from-vram]
                      [--pytorch-deterministic] [--disable-server-log]
                      [--debug-mode] [--is-windows-embedded-python]
                      [--disable-server-info] [--multi-user] [--share]
                      [--preset PRESET] [--disable-preset-selection]
                      [--language LANGUAGE]
                      [--disable-offload-from-vram] [--theme THEME]
                      [--disable-image-log] [--disable-analytics]
                      [--disable-metadata] [--disable-preset-download]
                      [--disable-enhance-output-sorting]
                      [--enable-auto-describe-image]
                      [--always-download-new-model]
                      [--rebuild-hash-cache [CPU_NUM_THREADS]]
```
</details>

## 多言語対応

`language/` フォルダに JSON を置いて `--language <名前>` で UI を翻訳できます。

```json
// language/jp.json の例
{
  "Generate": "生成",
  "Input Image": "入力画像",
  "Advanced": "詳細設定"
}
```

---

## mashb1t 1-Up Edition の追加機能

<details>
<summary>mashb1t による PR 一覧 (90+ マージ済み)</summary>

主な追加機能:
- プロンプト翻訳
- LCM リアルタイムキャンバスペインティング
- SDXL Turbo プリセット (DreamShaperXL_Turbo)
- プレビュー画像の軽量化
- Inpaint マスク自動生成 (rembg + セグメンテーション)
- メタデータ対応 (Fooocus JSON / A1111 プレーンテキスト)
- スタイルプレビュー (マウスオーバー)
- 画像拡張子対応 (PNG/JPG/WebP)
- UI プリセット切り替え
- 画像アップロード時の自動キャプション

全 PR: [mashb1t's PRs](https://github.com/lllyasviel/Fooocus/pulls/mashb1t)
</details>

## フォーク一覧

- [fenneishi/Fooocus-Control](https://github.com/fenneishi/Fooocus-Control)
- [runew0lf/RuinedFooocus](https://github.com/runew0lf/RuinedFooocus)
- [MoonRide303/Fooocus-MRE](https://github.com/MoonRide303/Fooocus-MRE)
- [metercai/SimpleSDXL](https://github.com/metercai/SimpleSDXL)
- [mashb1t/Fooocus](https://github.com/mashb1t/Fooocus)

## 謝辞

- スタイル追加の貢献者の皆さん
- ベースコード: [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) + [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## 更新ログ

[update_log.md](update_log.md)
