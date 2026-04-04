import json
import os
import unittest


class TestAnimaPreset(unittest.TestCase):
    def test_anima_preview2_preset_uses_safe_defaults(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        preset_path = os.path.join(repo_root, 'presets', 'anima_preview2.json')

        with open(preset_path, encoding='utf-8') as f:
            preset = json.load(f)

        self.assertEqual(preset['default_model'], 'anima-preview2.safetensors')
        self.assertEqual(preset['default_vae'], 'qwen_image_vae.safetensors')
        self.assertEqual(preset['default_sampler'], 'euler')
        self.assertEqual(preset['default_scheduler'], 'simple')
        self.assertEqual(preset['default_cfg_scale'], 4.0)
        self.assertEqual(preset['default_performance'], 'Speed')
        self.assertEqual(preset['default_overwrite_step'], 20)
        self.assertEqual(preset['default_aspect_ratio'], '1024*1024')
        self.assertEqual(preset['default_styles'], [])
        self.assertEqual(preset['default_refiner'], 'None')

    def test_anima_preview2_preset_keeps_required_downloads(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        preset_path = os.path.join(repo_root, 'presets', 'anima_preview2.json')

        with open(preset_path, encoding='utf-8') as f:
            preset = json.load(f)

        self.assertIn('anima-preview2.safetensors', preset['checkpoint_downloads'])
        self.assertIn('qwen_image_vae.safetensors', preset['vae_downloads'])


if __name__ == '__main__':
    unittest.main()
