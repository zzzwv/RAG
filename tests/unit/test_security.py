import logging
import unittest

from rag_app.logging_config import SensitiveDataFilter
from rag_app.security.admin_auth import hash_admin_password, verify_admin_password


class SecurityTests(unittest.TestCase):
    def test_admin_password_hash_is_salted_and_verifiable(self):
        first = hash_admin_password("secret")
        second = hash_admin_password("secret")
        self.assertNotEqual(first, second)
        self.assertTrue(verify_admin_password("secret", first))
        self.assertFalse(verify_admin_password("wrong", first))

    def test_log_filter_redacts_authorization_and_api_key(self):
        record = logging.LogRecord(
            "test", logging.INFO, __file__, 1,
            "Authorization: Bearer abc api_key=very-secret", (), None,
        )
        self.assertTrue(SensitiveDataFilter().filter(record))
        self.assertNotIn("very-secret", record.msg)
        self.assertNotIn("Bearer abc", record.msg)


if __name__ == "__main__":
    unittest.main()
