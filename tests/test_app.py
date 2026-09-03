import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mac_dit.app import COMMANDS, main
from mac_dit.hardware import _supports_metal


class UnifiedCliTests(unittest.TestCase):
    def test_root_help_lists_core_commands(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(["--help"])

        self.assertIsNone(result)
        self.assertIn("generate", output.getvalue())
        self.assertIn("mlx-generate", output.getvalue())

    def test_subcommand_arguments_are_forwarded_unchanged(self):
        received = []

        def handler(arguments):
            received.extend(arguments)
            return "ok"

        with patch.dict(COMMANDS, {"generate": handler}, clear=True):
            result = main(["generate", "--steps", "7", "--seed", "42"])

        self.assertEqual(result, "ok")
        self.assertEqual(received, ["--steps", "7", "--seed", "42"])

    def test_metal_detection_supports_old_and_new_system_profile_fields(self):
        self.assertTrue(
            _supports_metal({"spdisplays_metal": "spdisplays_supported"})
        )
        self.assertTrue(
            _supports_metal(
                {"spdisplays_mtlgpufamilysupport": "spdisplays_metal3"}
            )
        )
        self.assertFalse(_supports_metal({}))


if __name__ == "__main__":
    unittest.main()
