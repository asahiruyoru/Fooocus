import html
import translators
from functools import lru_cache


@lru_cache(maxsize=32, typed=False)
def translate2en(text, element):
    if not text:
        return text

    try:
        result = translators.translate_text(text, to_language='en')
        print(f'[Parameters] Translated {element}: {result}')
        return result
    except Exception as e:
        print(f'[Parameters] Error during translation of {element}: {e}')
        return text


@lru_cache(maxsize=512, typed=False)
def _translate_chunk_to_ja(text):
    if not text or not text.strip():
        return text
    try:
        return translators.translate_text(text, to_language='ja')
    except Exception as e:
        print(f'[Preview] Error during translation of "{text}": {e}')
        return text


PREVIEW_COLORS = [
    '#e74c3c', '#3498db', '#2ecc71', '#f39c12',
    '#9b59b6', '#1abc9c', '#e67e22', '#2c3e50',
    '#d35400', '#16a085', '#27ae60', '#2980b9',
    '#8e44ad', '#c0392b', '#f1c40f', '#7f8c8d',
]


def preview_translation_html(text):
    if not text or not text.strip():
        return ''

    chunks = [c.strip() for c in text.split(',')]
    chunks = [c for c in chunks if c]

    rows = []
    for i, chunk in enumerate(chunks):
        color = PREVIEW_COLORS[i % len(PREVIEW_COLORS)]
        translated = _translate_chunk_to_ja(chunk)
        rows.append(
            f'<div style="display:flex;gap:12px;margin:3px 0;align-items:baseline;">'
            f'<span style="color:{color};font-weight:600;flex:1;">{html.escape(chunk)}</span>'
            f'<span style="color:#888;flex:0 0 auto;">→</span>'
            f'<span style="color:{color};flex:1;">{html.escape(translated)}</span>'
            f'</div>'
        )

    return (
        '<div style="padding:8px 12px;border:1px solid #444;border-radius:6px;'
        'background:rgba(128,128,128,0.08);font-size:0.95em;">'
        + ''.join(rows)
        + '</div>'
    )
