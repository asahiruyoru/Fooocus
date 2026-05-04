import torch

from ldm_patched.modules.conds import CONDRegular
from modules import sample_hijack


def test_clip_separate_after_preparation_keeps_anima_conditioning(monkeypatch):
    class FakeAnimaModel:
        pass

    monkeypatch.setattr(sample_hijack, "AnimaModel", FakeAnimaModel)

    cross_attn = torch.randn(1, 32, 4096)
    pooled = torch.randn(1, 1024)
    cond = [{
        "model_conds": {"c_crossattn": CONDRegular(cross_attn)},
        "pooled_output": pooled,
    }]

    separated = sample_hijack.clip_separate_after_preparation(
        cond,
        target_model=FakeAnimaModel(),
    )

    assert len(separated) == 1
    assert torch.equal(separated[0]["model_conds"]["c_crossattn"].cond, cross_attn)
    assert torch.equal(separated[0]["pooled_output"], pooled)
    assert separated[0]["model_conds"]["c_crossattn"].cond is not cross_attn
    assert separated[0]["pooled_output"] is not pooled
