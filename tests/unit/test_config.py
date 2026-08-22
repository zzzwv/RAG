import json
import tempfile
import unittest
from pathlib import Path

from rag_app.config import AppSettings


class ConfigTests(unittest.TestCase):
    def test_json_override_is_loaded_and_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "settings.json")
            path.write_text(json.dumps({"memory_turns": 6, "normal_chunk_size": 900}), encoding="utf-8")
            settings = AppSettings.load(path)
        self.assertEqual(settings.memory_turns, 6)
        self.assertEqual(settings.normal_chunk_size, 900)

    def test_invalid_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            AppSettings(normal_chunk_size=100, normal_chunk_overlap=100).validate()


if __name__ == "__main__":
    unittest.main()
