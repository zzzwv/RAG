from pathlib import Path

from cryptography.fernet import Fernet

from rag_app.security.secrets import EncryptedSecretStore


def test_secret_file_is_encrypted_and_round_trips(tmp_path: Path):
    path = tmp_path / "secrets.enc"
    store = EncryptedSecretStore(path, key=Fernet.generate_key())
    store.write({"api_key": "super-secret-value"})
    assert b"super-secret-value" not in path.read_bytes()
    assert store.read() == {"api_key": "super-secret-value"}
