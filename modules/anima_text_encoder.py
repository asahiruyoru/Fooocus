"""
Anima Preview2 Text Encoder - Qwen3 integration for Fooocus.

Loads the Qwen3-0.6B model from safetensors and produces text embeddings
for the Anima DiT model's cross-attention conditioning.
"""
import os
import torch
import ldm_patched.modules.model_management as model_management

# Global cache for the loaded encoder
_anima_text_encoder = None


class AnimaTextEncoder:
    """Wraps a Qwen3-0.6B text encoder for Anima conditioning."""

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = model_management.text_encoder_device()
        self.offload_device = model_management.text_encoder_offload_device()
        self.dtype = model_management.text_encoder_dtype(self.device)
        self._load(model_path)

    def _load(self, model_path):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from safetensors.torch import load_file

        print(f"[AnimaTextEncoder] Loading Qwen3 text encoder from: {model_path}")

        # Load tokenizer from HuggingFace hub (cached after first download)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-0.6B",
                trust_remote_code=True,
            )
            print("[AnimaTextEncoder] Loaded Qwen3 tokenizer from HuggingFace hub")
        except Exception as e:
            print(f"[AnimaTextEncoder] Failed to load tokenizer from hub: {e}")
            print("[AnimaTextEncoder] Falling back to basic tokenizer")
            self.tokenizer = None

        # Load model architecture using AutoModel config, then load our weights
        try:
            # First, load the model structure from HuggingFace
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen3-0.6B",
                torch_dtype=self.dtype,
                trust_remote_code=True,
            )

            # If we have a local safetensors file, load those weights instead
            if os.path.exists(model_path):
                print(f"[AnimaTextEncoder] Loading weights from local file: {model_path}")
                state_dict = load_file(model_path)
                # Try to load the state dict (it may use different key prefixes)
                try:
                    model.load_state_dict(state_dict, strict=False)
                except Exception as e2:
                    print(f"[AnimaTextEncoder] Non-strict load completed with: {e2}")

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
        """Encode text into hidden states for Anima conditioning.

        Returns:
            tuple: (hidden_states, token_ids) where
                - hidden_states: tensor of shape (1, seq_len, hidden_dim)
                - token_ids: tensor of token IDs (for the LLM adapter)
        """
        if self.model is None or self.tokenizer is None:
            return self._encode_empty()

        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                padding="max_length",
                truncation=True,
            )

            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", None)

            # Move model to compute device
            self.model.to(self.device)

            # Get hidden states
            input_ids = input_ids.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            # Use the last hidden state as text embeddings
            hidden_states = outputs.hidden_states[-1]

            # Move model back to offload device
            self.model.to(self.offload_device)

            # Clamp token IDs to fit within the LLM adapter's embedding table (32128 entries).
            # Qwen3's pad_token_id (151643) exceeds this range and would cause CUDA index errors.
            # Replace out-of-range IDs with 0 (will be masked by attention_mask anyway).
            clamped_ids = input_ids.cpu()
            clamped_ids[clamped_ids >= 32128] = 0

            # Return hidden states and token IDs (both on CPU)
            return hidden_states.cpu(), clamped_ids

        except Exception as e:
            print(f"[AnimaTextEncoder] Encoding failed: {e}")
            if self.model is not None:
                self.model.to(self.offload_device)
            return self._encode_empty()

    def _encode_empty(self):
        """Return empty conditioning as fallback."""
        # Anima uses 1024-dim cross-attention
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
