from __future__ import annotations

import json
import os
from pathlib import Path


class EncryptedSecretStore:
    def __init__(self, path: str | Path = "config/secrets.enc", key: str | bytes | None = None) -> None:
        self.path = Path(path)
        supplied = key or os.getenv("RAG_MASTER_KEY")
        if not supplied:
            raise RuntimeError("缺少 RAG_MASTER_KEY，无法读取加密配置")
        from cryptography.fernet import Fernet

        self.fernet = Fernet(supplied.encode() if isinstance(supplied, str) else supplied)

    def read(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        decrypted = self.fernet.decrypt(self.path.read_bytes())
        return json.loads(decrypted.decode("utf-8"))

    def write(self, values: dict[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(values, ensure_ascii=False).encode("utf-8")
        temporary = self.path.with_suffix(".tmp")
        temporary.write_bytes(self.fernet.encrypt(payload))
        temporary.replace(self.path)

    def get(self, key: str, default: str = "") -> str:
        return self.read().get(key, default)
