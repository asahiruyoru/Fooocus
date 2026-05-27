import json
import os
import unittest


class TestAnimaPreset(unittest.TestCase):
    def test_anima_base_v1_preset_uses_safe_defaults(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        preset_path = os.path.join(repo_root, 'presets', 'anima_base_v1.json')

        with open(preset_path, encoding='utf-8') as f:
            preset = json.load(f)

        self.assertEqual(preset['default_model'], 'anima-base-v1.0.safetensors')
        self.assertEqual(preset['default_vae'], 'qwen_image_vae.safetensors')
        self.assertEqual(preset['default_sampler'], 'euler_ancestral')
        self.assertEqual(preset['default_scheduler'], 'simple')
        self.assertEqual(preset['default_cfg_scale'], 4.5)
        self.assertEqual(preset['default_performance'], 'Quality')
        self.assertEqual(preset['default_advanced_checkbox'], True)
        self.assertEqual(preset['default_image_number'], 32)
        self.assertEqual(preset['default_save_metadata_to_images'], True)
        self.assertEqual(preset['default_metadata_scheme'], 'a1111')
        self.assertEqual(
            preset['default_prompt'],
            'masterpiece, best quality, score_7, highres, safe, 1girl, solo, '
            'looking at viewer, smile, long hair, detailed eyes, detailed face, '
            'clean lines, smooth shading, soft lighting',
        )
        self.assertEqual(
            preset['default_prompt_negative'],
            'worst quality, low quality, score_1, score_2, score_3, artist name',
        )
        self.assertEqual(preset['default_overwrite_step'], -1)
        self.assertEqual(preset['default_aspect_ratio'], '1344*1344')
        self.assertEqual(preset['default_styles'], [])
        self.assertEqual(preset['default_refiner'], 'None')

    def test_anima_base_v1_preset_keeps_required_downloads(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        preset_path = os.path.join(repo_root, 'presets', 'anima_base_v1.json')

        with open(preset_path, encoding='utf-8') as f:
            preset = json.load(f)

        self.assertIn('anima-base-v1.0.safetensors', preset['checkpoint_downloads'])
        self.assertIn('qwen_image_vae.safetensors', preset['vae_downloads'])
        self.assertIn('qwen_3_06b_base.safetensors', preset['clip_downloads'])

    def test_hassaku_anima_v01_preset_uses_safe_defaults(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        preset_path = os.path.join(repo_root, 'presets', 'hassaku_anima_v01.json')

        with open(preset_path, encoding='utf-8') as f:
            preset = json.load(f)

        self.assertEqual(preset['default_model'], 'hassakuAnima_v01.safetensors')
        self.assertEqual(preset['default_vae'], 'qwen_image_vae.safetensors')
        self.assertEqual(preset['default_sampler'], 'euler_ancestral')
        self.assertEqual(preset['default_scheduler'], 'simple')
        self.assertEqual(preset['default_performance'], 'Quality')

    def test_hassaku_anima_v01_preset_keeps_required_downloads(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        preset_path = os.path.join(repo_root, 'presets', 'hassaku_anima_v01.json')

        with open(preset_path, encoding='utf-8') as f:
            preset = json.load(f)

        self.assertIn('hassakuAnima_v01.safetensors', preset['checkpoint_downloads'])
        self.assertIn('qwen_image_vae.safetensors', preset['vae_downloads'])
        self.assertIn('qwen_3_06b_base.safetensors', preset['clip_downloads'])


if __name__ == '__main__':
    unittest.main()
