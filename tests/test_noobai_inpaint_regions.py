import numpy as np
import pytest

from modules.image_input_selection import merge_noobai_inpaint_inputs
from modules.noobai_inpaint_regions import build_noobai_inpaint_region_mask, parse_noobai_inpaint_regions


def test_parse_noobai_inpaint_regions_supports_top_level_normalized_flag():
    regions = parse_noobai_inpaint_regions(
        '{"normalized": true, "regions": [{"cx": 0.5, "cy": 0.5, "width": 0.2, "height": 0.1, "angle": 15}]}'
    )

    assert len(regions) == 1
    assert regions[0]['normalized'] is True
    assert regions[0]['angle'] == 15.0


def test_build_noobai_inpaint_region_mask_combines_multiple_rotated_rectangles():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    raw_regions = """
    [
      {"cx": 30, "cy": 40, "width": 20, "height": 10, "angle": 0},
      {"cx": 80, "cy": 50, "width": 24, "height": 12, "angle": 45}
    ]
    """

    mask, region_count = build_noobai_inpaint_region_mask(image.shape, raw_regions)

    assert region_count == 2
    assert mask[40, 30] == 255
    assert mask[50, 80] == 255
    assert mask[0, 0] == 0


def test_build_noobai_inpaint_region_mask_supports_normalized_regions():
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    raw_regions = '{"normalized": true, "regions": [[0.5, 0.5, 0.2, 0.25, 0]]}'

    mask, region_count = build_noobai_inpaint_region_mask(image.shape, raw_regions)

    assert region_count == 1
    assert mask[40, 60] == 255
    assert mask[5, 5] == 0


def test_parse_noobai_inpaint_regions_rejects_invalid_json():
    with pytest.raises(ValueError, match='Failed to parse NoobAI inpaint regions JSON'):
        parse_noobai_inpaint_regions('{"regions": [}')


def test_merge_noobai_inpaint_inputs_keeps_rotated_regions():
    merged = merge_noobai_inpaint_inputs(
        ['noobai_inpaint'],
        {'image': np.zeros((8, 8, 3), dtype=np.uint8), 'mask': np.zeros((8, 8, 3), dtype=np.uint8)},
        'detail face',
        '[{"cx":4,"cy":4,"width":2,"height":2,"angle":30}]',
        None,
        '',
        [],
    )

    assert merged is not None
    assert merged['regions'] == '[{"cx":4,"cy":4,"width":2,"height":2,"angle":30}]'
    assert merged['additional_prompt'] == 'detail face'
