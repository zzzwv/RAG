from __future__ import annotations

import base64
import hashlib
import hmac
import os

_N = 2**14
_R = 8
_P = 1


def hash_admin_password(password: str) -> str:
    if not password:
        raise ValueError("运维口令不能为空")
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    return "scrypt${}${}${}${}${}".format(
        _N,
        _R,
        _P,
        base64.urlsafe_b64encode(salt).decode(),
        base64.urlsafe_b64encode(digest).decode(),
    )


def verify_admin_password(password: str, encoded: str) -> bool:
    try:
        algorithm, n, r, p, salt, expected = encoded.split("$", 5)
        if algorithm != "scrypt":
            return False
        digest = hashlib.scrypt(
            password.encode(),
            salt=base64.urlsafe_b64decode(salt.encode()),
            n=int(n), r=int(r), p=int(p), dklen=32,
        )
        return hmac.compare_digest(digest, base64.urlsafe_b64decode(expected.encode()))
    except (ValueError, TypeError):
        return False
