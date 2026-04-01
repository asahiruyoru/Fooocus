"""
Anima Preview2 Text Encoder - Dual tokenizer for Fooocus.

Uses Qwen3-0.6B to produce hidden states (source embeddings) and
T5 tokenizer to produce token IDs for the LLM adapter's Embedding(32128).
The LLM adapter cross-attends between the T5 token embeddings and Qwen3
hidden states to produce the final conditioning for the DiT model.
"""
import os
import torch
import ldm_patched.modules.model_management as model_management

# Global cache for the loaded encoder
_anima_text_encoder = None


class AnimaTextEncoder:
    """Dual-tokenizer encoder: Qwen3 for hidden states, T5 for token IDs."""

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.qwen_tokenizer = None
        self.t5_tokenizer = None
        self.device = model_management.text_encoder_device()
        self.offload_device = model_management.text_encoder_offload_device()
        self.dtype = model_management.text_encoder_dtype(self.device)
        self._load(model_path)

    def _load(self, model_path):
        from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer
        from safetensors.torch import load_file

        print(f"[AnimaTextEncoder] Loading Qwen3 text encoder from: {model_path}")

        # Load Qwen3 tokenizer (for hidden state generation)
        try:
            self.qwen_tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-0.6B",
                trust_remote_code=True,
            )
            print("[AnimaTextEncoder] Loaded Qwen3 tokenizer")
        except Exception as e:
            print(f"[AnimaTextEncoder] Failed to load Qwen3 tokenizer: {e}")

        # Load T5 tokenizer (for LLM adapter token IDs, vocab=32128)
        try:
            self.t5_tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")
            print("[AnimaTextEncoder] Loaded T5 tokenizer (vocab=32128)")
        except Exception as e:
            print(f"[AnimaTextEncoder] Failed to load T5 tokenizer: {e}")

        # Load Qwen3 model
        try:
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen3-0.6B",
                torch_dtype=self.dtype,
                trust_remote_code=True,
            )

            if os.path.exists(model_path):
                print(f"[AnimaTextEncoder] Loading weights from: {model_path}")
                state_dict = load_file(model_path)
                try:
                    model.load_state_dict(state_dict, strict=False)
                except Exception as e2:
                    print(f"[AnimaTextEncoder] Non-strict load: {e2}")

            model.eval()
            model.to(self.offload_device)
            self.model = model
            print("[AnimaTextEncoder] Qwen3 model loaded successfully")

        except Exception as e:
            print(f"[AnimaTextEncoder] Failed to load Qwen3 model: {e}")
            self.model = None

    @torch.no_grad()
    @torch.inference_mode()
    def encode(self, text, max_length=256):
        """Encode text using dual tokenizer approach.

        Returns:
            tuple: (hidden_states, token_ids) where
                - hidden_states: [1, seq_len, 1024] from Qwen3 last layer
                - token_ids: [1, seq_len] T5 token IDs for LLM adapter
        """
        if self.model is None or self.qwen_tokenizer is None:
            return self._encode_empty()

        try:
            # Step 1: Qwen3 tokenizer -> model -> hidden states
            qwen_inputs = self.qwen_tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                padding="max_length",
                truncation=True,
            )
            input_ids = qwen_inputs["input_ids"].to(self.device)
            attention_mask = qwen_inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            self.model.to(self.device)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[-1]
            self.model.to(self.offload_device)

            # Step 2: T5 tokenizer -> token IDs for LLM adapter embedding(32128)
            if self.t5_tokenizer is not None:
                t5_inputs = self.t5_tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                )
                token_ids = t5_inputs["input_ids"]
            else:
                # Fallback: clamped Qwen3 IDs (worse quality)
                token_ids = input_ids.cpu()
                token_ids[token_ids >= 32128] = 0

            return hidden_states.cpu(), token_ids.cpu()

        except Exception as e:
            print(f"[AnimaTextEncoder] Encoding failed: {e}")
            if self.model is not None:
                self.model.to(self.offload_device)
            return self._encode_empty()

    def _encode_empty(self):
        """Return empty conditioning as fallback."""
        hidden_states = torch.zeros(1, 256, 1024)
        token_ids = torch.zeros(1, 256, dtype=torch.long)
        return hidden_states, token_ids


def get_anima_text_encoder(model_path=None):
    """Get or create the global Anima text encoder instance."""
    global _anima_text_encoder

    if _anima_text_encoder is not None:
        return _anima_text_encoder

    if model_path is None:
        # Try to find the Qwen3 model file in common locations
        import modules.config as config
        clip_dirs = []

        # Check path_clip if it exists
        if hasattr(config, 'path_clip'):
            clip_dirs.append(config.path_clip)

        # Also check models/clip/ relative to the Fooocus root
        fooocus_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        clip_dirs.append(os.path.join(fooocus_root, 'models', 'clip'))

        # Also check path_vae parent's sibling 'clip' directory
        if hasattr(config, 'path_vae'):
            vae_dir = config.path_vae if isinstance(config.path_vae, str) else config.path_vae[0] if isinstance(config.path_vae, list) else None
            if vae_dir:
                clip_dirs.append(os.path.join(os.path.dirname(vae_dir), 'clip'))

        for clip_dir in clip_dirs:
            candidate = os.path.join(clip_dir, 'qwen_3_06b_base.safetensors')
            if os.path.exists(candidate):
                model_path = candidate
                break

    if model_path is None or not os.path.exists(model_path):
        print("[AnimaTextEncoder] WARNING: Qwen3 text encoder not found. "
              "Text conditioning will be empty. "
              "Please download qwen_3_06b_base.safetensors to models/clip/")
        # Return a stub encoder that produces empty conditioning
        _anima_text_encoder = AnimaTextEncoderStub()
        return _anima_text_encoder

    _anima_text_encoder = AnimaTextEncoder(model_path)
    return _anima_text_encoder


class AnimaTextEncoderStub:
    """Stub encoder that produces empty conditioning when Qwen3 model is not available."""

    @torch.no_grad()
    def encode(self, text, max_length=256):
        hidden_states = torch.zeros(1, 256, 1024)
        token_ids = torch.zeros(1, 256, dtype=torch.long)
        return hidden_states, token_ids
