import sys
import unittest
from unittest.mock import MagicMock, patch
from src.chatbot.package_manager import PackageManager

class TestPackageManager(unittest.TestCase):
    def setUp(self):
        self.pm = PackageManager(max_retries=2)

    @patch('subprocess.run')
    def test_loop_prevention(self, mock_run):
        # Mock failure
        mock_run.return_value = MagicMock(returncode=1, stderr="Not found")
        
        # Attempt 1
        print("Attempt 1...")
        self.pm.install(["test-pkg"])
        
        # Attempt 2
        print("Attempt 2...")
        self.pm.install(["test-pkg"])
        
        # Attempt 3 (Should be blocked)
        print("Attempt 3...")
        result = self.pm.install(["test-pkg"])
        
        print(f"Result 3: {result}")
        self.assertTrue(result.get("skipped"), "Should have skipped due to max retries")
        self.assertIn("Failed 2 times previously", result["message"])

    @patch('subprocess.run')
    def test_smart_fallback(self, mock_run):
        # Mock failure for specific version, success for generic
        def side_effect(cmd, **kwargs):
            if "numpy==9.9.9" in cmd:
                return MagicMock(returncode=1, stderr="No matching distribution")
            if "numpy" in cmd:
                return MagicMock(returncode=0, stdout="Successfully installed numpy")
            return MagicMock(returncode=1)
            
        mock_run.side_effect = side_effect
        
        # Attempt 1: Strict version
        print("\nAttempt 1 (strict)...")
        res1 = self.pm.install(["numpy==9.9.9"])
        self.assertFalse(res1["success"])
        
        # Attempt 2: Retry (Should fallback to 'numpy')
        print("Attempt 2 (fallback)...")
        res2 = self.pm.install(["numpy==9.9.9"])
        
        print(f"Result 2: {res2}")
        self.assertTrue(res2["success"])
        self.assertIn("Retrying 'numpy' without version constraints", res2["message"])

if __name__ == '__main__':
    unittest.main()
