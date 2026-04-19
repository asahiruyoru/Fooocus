import json
import math

import cv2
import numpy as np


def _coerce_float(value, field_name, region_index):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f'NoobAI inpaint region #{region_index + 1} has an invalid "{field_name}" value: {value!r}.'
        ) from exc

    if not math.isfinite(result):
        raise ValueError(
            f'NoobAI inpaint region #{region_index + 1} has a non-finite "{field_name}" value.'
        )

    return result


def _parse_single_region(raw_region, region_index, default_normalized=False):
    if isinstance(raw_region, (list, tuple)):
        if len(raw_region) != 5:
            raise ValueError(
                'NoobAI inpaint regions must use either an object '
                'or a five-item array: [cx, cy, width, height, angle].'
            )

        center_x, center_y, width, height, angle = raw_region
        normalized = default_normalized
    elif isinstance(raw_region, dict):
        normalized = bool(raw_region.get('normalized', default_normalized))
        angle = raw_region.get('angle', raw_region.get('rotation', raw_region.get('degrees', 0.0)))
        width = raw_region.get('width', raw_region.get('w'))
        height = raw_region.get('height', raw_region.get('h'))

        if 'cx' in raw_region or 'cy' in raw_region:
            center_x = raw_region.get('cx')
            center_y = raw_region.get('cy')
        elif 'center_x' in raw_region or 'center_y' in raw_region:
            center_x = raw_region.get('center_x')
            center_y = raw_region.get('center_y')
        elif 'x' in raw_region or 'y' in raw_region:
            x = _coerce_float(raw_region.get('x'), 'x', region_index)
            y = _coerce_float(raw_region.get('y'), 'y', region_index)
            width = _coerce_float(width, 'width', region_index)
            height = _coerce_float(height, 'height', region_index)
            center_x = x + width / 2.0
            center_y = y + height / 2.0
        else:
            center_x = raw_region.get('centerX')
            center_y = raw_region.get('centerY')
    else:
        raise ValueError(
            f'NoobAI inpaint region #{region_index + 1} must be an object or array, '
            f'but got {type(raw_region).__name__}.'
        )

    center_x = _coerce_float(center_x, 'cx', region_index)
    center_y = _coerce_float(center_y, 'cy', region_index)
    width = _coerce_float(width, 'width', region_index)
    height = _coerce_float(height, 'height', region_index)
    angle = _coerce_float(angle, 'angle', region_index)

    if width <= 0 or height <= 0:
        raise ValueError(
            f'NoobAI inpaint region #{region_index + 1} must have positive width and height.'
        )

    return {
        'cx': center_x,
        'cy': center_y,
        'width': width,
        'height': height,
        'angle': angle,
        'normalized': normalized,
    }


def parse_noobai_inpaint_regions(raw_regions):
    if raw_regions is None:
        return []

    if not isinstance(raw_regions, str):
        raise ValueError('NoobAI inpaint regions must be provided as a JSON string.')

    stripped = raw_regions.strip()
    if stripped == '':
        return []

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f'Failed to parse NoobAI inpaint regions JSON: {exc.msg} (line {exc.lineno}, column {exc.colno}).'
        ) from exc

    default_normalized = False
    if isinstance(parsed, dict):
        default_normalized = bool(parsed.get('normalized', False))
        parsed = parsed.get('regions')

    if not isinstance(parsed, list):
        raise ValueError(
            'NoobAI inpaint regions JSON must be either an array of regions or '
            'an object with a "regions" array.'
        )

    return [
        _parse_single_region(region, region_index, default_normalized=default_normalized)
        for region_index, region in enumerate(parsed)
    ]


def build_noobai_inpaint_region_mask(image_shape, raw_regions):
    image_height, image_width = image_shape[:2]
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    regions = parse_noobai_inpaint_regions(raw_regions)

    for region in regions:
        scale_x = image_width if region['normalized'] else 1.0
        scale_y = image_height if region['normalized'] else 1.0
        rect = (
            (region['cx'] * scale_x, region['cy'] * scale_y),
            (region['width'] * scale_x, region['height'] * scale_y),
            region['angle'],
        )
        points = np.round(cv2.boxPoints(rect)).astype(np.int32)
        cv2.fillConvexPoly(mask, points, 255)

    return mask, len(regions)
