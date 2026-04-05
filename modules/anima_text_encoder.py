"""
Anima Preview2 text encoder helpers.

Prefer the bundled ComfyUI reference implementation so that Anima uses the
same local tokenizers and Qwen3 encoder path as its native workflow. Fall
back to the older Hugging Face based path only if the reference loader fails.
"""
import os
import sys
from pathlib import Path

import torch
import ldm_patched.modules.model_management as model_management


_anima_text_encoder = None


class AnimaTextEncoder:
    """Encode Anima prompts into Qwen3 hidden states plus T5 token metadata."""

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.qwen_tokenizer = None
        self.t5_tokenizer = None
        self.reference_model = None
        self.reference_tokenizer = None
        self.backend = None
        self.device = model_management.text_encoder_device()
        self.offload_device = model_management.text_encoder_offload_device()
        self.dtype = model_management.text_encoder_dtype(self.device)
        self._load(model_path)

    def _fooocus_root(self):
        return Path(__file__).resolve().parents[1]

    def _candidate_comfy_roots(self):
        roots = []

        env_root = os.environ.get("ANIMA_COMFY_ROOT")
        if env_root:
            roots.append(Path(env_root))

        roots.extend([
            self._fooocus_root() / "comfyui_tmp",
            Path("/content/ComfyUI"),
            Path("/content/comfyui"),
            Path("/content/comfy"),
        ])

        seen = set()
        ordered = []
        for root in roots:
            root_str = str(root.resolve()) if root.exists() else str(root)
            if root_str in seen:
                continue
            seen.add(root_str)
            ordered.append(root)
        return ordered

    def _ensure_comfyui_tmp_path(self):
        for comfy_root in self._candidate_comfy_roots():
            if not (comfy_root / "comfy" / "sd.py").exists():
                continue
            comfy_root_str = str(comfy_root)
            if comfy_root_str not in sys.path:
                sys.path.insert(0, comfy_root_str)
            return comfy_root
        return None

    def _reset_comfy_modules(self, comfy_root):
        comfy_root_str = str(comfy_root)
        for name in list(sys.modules.keys()):
            if not (name == "comfy" or name.startswith("comfy.") or name == "comfy_aimdo" or name.startswith("comfy_aimdo.")):
                continue
            module = sys.modules.get(name)
            module_file = getattr(module, "__file__", "") or ""
            if module_file and module_file.startswith(comfy_root_str):
                continue
            sys.modules.pop(name, None)

    def _load(self, model_path):
        if self._load_comfy_reference(model_path):
            return
        self._load_huggingface_fallback(model_path)

    def _load_comfy_reference(self, model_path):
        if not os.path.exists(model_path):
            return False

        try:
            comfy_root = self._ensure_comfyui_tmp_path()
            if comfy_root is None:
                print("[AnimaTextEncoder] ComfyUI reference not found in candidate roots")
                return False

            print(f"[AnimaTextEncoder] Using ComfyUI root: {comfy_root}")
            self._reset_comfy_modules(comfy_root)

            from safetensors.torch import load_file
            import comfy.text_encoders.anima as comfy_anima

            print(f"[AnimaTextEncoder] Loading ComfyUI reference encoder from: {model_path}")
            state_dict = load_file(model_path)
            self.reference_tokenizer = comfy_anima.AnimaTokenizer(tokenizer_data={})
            self.reference_model = comfy_anima.AnimaTEModel(
                device=self.offload_device,
                dtype=self.dtype,
                model_options={},
            )
            self.reference_model.load_sd(state_dict)
            self.reference_model.eval()
            self.model = self.reference_model
            self.qwen_tokenizer = getattr(self.reference_tokenizer, "qwen3_06b", None)
            self.t5_tokenizer = getattr(self.reference_tokenizer, "t5xxl", None)
            self.backend = "comfy_reference_direct"
            print("[AnimaTextEncoder] Loaded bundled ComfyUI reference tokenizer + Qwen3 encoder")
            return True
        except Exception as e:
            print(f"[AnimaTextEncoder] ComfyUI reference load failed: {e}")
            self.reference_model = None
            self.reference_tokenizer = None
            return False

    def _load_huggingface_fallback(self, model_path):
        from transformers import (
            AutoModel,
            AutoModelForCausalLM,
            AutoTokenizer,
            Qwen2Tokenizer,
            T5Tokenizer,
            T5TokenizerFast,
        )
        from safetensors.torch import load_file

        print(f"[AnimaTextEncoder] Loading Hugging Face fallback encoder from: {model_path}")

        qwen_tokenizer_dir = (
            self._fooocus_root() / "comfyui_tmp" / "comfy" / "text_encoders" / "qwen25_tokenizer"
        )
        t5_tokenizer_dir = (
            self._fooocus_root() / "comfyui_tmp" / "comfy" / "text_encoders" / "t5_tokenizer"
        )

        try:
            if qwen_tokenizer_dir.exists():
                self.qwen_tokenizer = Qwen2Tokenizer.from_pretrained(
                    str(qwen_tokenizer_dir),
                    trust_remote_code=True,
                )
                print("[AnimaTextEncoder] Loaded Qwen tokenizer from bundled tokenizer files")
            else:
                self.qwen_tokenizer = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen3-0.6B",
                    trust_remote_code=True,
                )
                print("[AnimaTextEncoder] Loaded Qwen3 tokenizer from Hugging Face")
        except Exception as e:
            print(f"[AnimaTextEncoder] Failed to load Qwen3 tokenizer: {e}")

        try:
            if t5_tokenizer_dir.exists():
                self.t5_tokenizer = T5TokenizerFast.from_pretrained(str(t5_tokenizer_dir))
                print("[AnimaTextEncoder] Loaded T5 tokenizer from bundled tokenizer files")
            else:
                self.t5_tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")
                print("[AnimaTextEncoder] Loaded T5 tokenizer from Hugging Face")
        except Exception as e:
            print(f"[AnimaTextEncoder] Failed to load T5 tokenizer: {e}")

        model = None
        state_dict = None
        if os.path.exists(model_path):
            print(f"[AnimaTextEncoder] Loading local weights from: {model_path}")
            state_dict = load_file(model_path)

        try:
            model = AutoModel.from_pretrained(
                "Qwen/Qwen3-0.6B",
                torch_dtype=self.dtype,
                trust_remote_code=True,
            )
            if state_dict is not None:
                model.load_state_dict(state_dict, strict=False)
            self.backend = "huggingface_base_model"
            print("[AnimaTextEncoder] Loaded Hugging Face fallback Qwen3 base model")
        except Exception as e:
            print(f"[AnimaTextEncoder] Base-model fallback failed: {e}")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    "Qwen/Qwen3-0.6B",
                    torch_dtype=self.dtype,
                    trust_remote_code=True,
                )
                if state_dict is not None:
                    model.load_state_dict(state_dict, strict=False)
                self.backend = "huggingface_causal_lm"
                print("[AnimaTextEncoder] Loaded Hugging Face fallback Qwen3 causal model")
            except Exception as e2:
                print(f"[AnimaTextEncoder] Failed to load Hugging Face fallback model: {e2}")
                self.model = None
                return

        model.eval()
        model.to(self.offload_device)
        self.model = model

    def _tokenize_unpadded(self, tokenizer, text, max_length):
        return tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=False,
        )

    @torch.no_grad()
    @torch.inference_mode()
    def encode(self, text, max_length=256):
        """Return (hidden_states, token_ids, token_weights)."""
        if self.reference_model is not None and self.reference_tokenizer is not None:
            return self._encode_with_comfy_reference(text)

        if self.model is None or self.qwen_tokenizer is None:
            return self._encode_empty(max_length=max_length)

        try:
            qwen_inputs = self._tokenize_unpadded(self.qwen_tokenizer, text, max_length)
            input_ids = qwen_inputs["input_ids"].to(self.device)
            attention_mask = qwen_inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            self.model.to(self.device)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            hidden_states = outputs.last_hidden_state.cpu()
            self.model.to(self.offload_device)

            if self.t5_tokenizer is not None:
                t5_inputs = self._tokenize_unpadded(self.t5_tokenizer, text, max_length)
                token_ids = t5_inputs["input_ids"].cpu()
            else:
                token_ids = input_ids.cpu()
                token_ids[token_ids >= 32128] = 0

            token_weights = torch.ones_like(token_ids, dtype=torch.float32)
            return hidden_states, token_ids, token_weights
        except Exception as e:
            print(f"[AnimaTextEncoder] Hugging Face fallback encode failed: {e}")
            if self.model is not None:
                self.model.to(self.offload_device)
            return self._encode_empty(max_length=max_length)

    def _encode_with_comfy_reference(self, text):
        try:
            tokens = self.reference_tokenizer.tokenize_with_weights(text)
            self.reference_model.to(self.device)
            hidden_states, _pooled, extra = self.reference_model.encode_token_weights(tokens)
            hidden_states = hidden_states.cpu()
            token_ids = extra.get("t5xxl_ids")
            token_weights = extra.get("t5xxl_weights")
            self.reference_model.to(self.offload_device)

            if token_ids is None:
                token_ids = torch.zeros(hidden_states.shape[1], dtype=torch.long)
            else:
                token_ids = token_ids.cpu()

            if token_weights is None:
                token_weights = torch.ones_like(token_ids, dtype=torch.float32)
            else:
                token_weights = token_weights.cpu()

            return hidden_states, token_ids, token_weights
        except Exception as e:
            print(f"[AnimaTextEncoder] ComfyUI reference encode failed: {e}")
            if self.reference_model is not None:
                self.reference_model.to(self.offload_device)
            return self._encode_empty()

    def _encode_empty(self, max_length=256):
        hidden_states = torch.zeros(1, max_length, 1024)
        token_ids = torch.zeros(max_length, dtype=torch.long)
        token_weights = torch.zeros(max_length, dtype=torch.float32)
        return hidden_states, token_ids, token_weights


def get_anima_text_encoder(model_path=None):
    """Get or create the global Anima text encoder instance."""
    global _anima_text_encoder

    if _anima_text_encoder is not None:
        return _anima_text_encoder

    if model_path is None:
        import modules.config as config

        clip_dirs = []
        if hasattr(config, "path_clip"):
            clip_dirs.append(config.path_clip)

        fooocus_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        clip_dirs.append(os.path.join(fooocus_root, "models", "clip"))

        if hasattr(config, "path_vae"):
            vae_dir = (
                config.path_vae
                if isinstance(config.path_vae, str)
                else config.path_vae[0] if isinstance(config.path_vae, list) else None
            )
            if vae_dir:
                clip_dirs.append(os.path.join(os.path.dirname(vae_dir), "clip"))

        for clip_dir in clip_dirs:
            candidate = os.path.join(clip_dir, "qwen_3_06b_base.safetensors")
            if os.path.exists(candidate):
                model_path = candidate
                break

    if model_path is None or not os.path.exists(model_path):
        print(
            "[AnimaTextEncoder] WARNING: Qwen3 text encoder not found. "
            "Text conditioning will be empty. "
            "Please download qwen_3_06b_base.safetensors to models/clip/"
        )
        _anima_text_encoder = AnimaTextEncoderStub()
        return _anima_text_encoder

    _anima_text_encoder = AnimaTextEncoder(model_path)
    return _anima_text_encoder


class AnimaTextEncoderStub:
    """Stub encoder that produces empty conditioning when Qwen3 is unavailable."""

    @torch.no_grad()
    def encode(self, text, max_length=256):
        hidden_states = torch.zeros(1, max_length, 1024)
        token_ids = torch.zeros(max_length, dtype=torch.long)
        token_weights = torch.zeros(max_length, dtype=torch.float32)
        return hidden_states, token_ids, token_weights
