from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_app.config import AppSettings
from rag_app.security.admin_auth import hash_admin_password
from rag_app.security.secrets import EncryptedSecretStore


def generate_key() -> None:
    from cryptography.fernet import Fernet

    print(Fernet.generate_key().decode())


def set_api_key() -> None:
    if not os.getenv("RAG_MASTER_KEY"):
        raise SystemExit("请先设置环境变量 RAG_MASTER_KEY（可由 generate-key 生成）")
    value = getpass.getpass("请输入 OpenAI 兼容接口 API key（本地接口可填 not-required）：")
    store = EncryptedSecretStore(ROOT / "config" / "secrets.enc")
    secrets = store.read()
    secrets["api_key"] = value
    store.write(secrets)
    print("API key 已加密保存到 config/secrets.enc")


def init_admin() -> None:
    first = getpass.getpass("请输入新的运维口令：")
    second = getpass.getpass("请再次输入：")
    if first != second:
        raise SystemExit("两次输入不一致")
    settings_path = ROOT / "config" / "settings.json"
    settings = AppSettings.load(settings_path)
    settings.admin_password_hash = hash_admin_password(first)
    settings.save(settings_path)
    print("运维口令已设置")


def main() -> None:
    parser = argparse.ArgumentParser(description="企业 RAG 安全配置工具")
    parser.add_argument("command", choices=["generate-key", "set-api-key", "init-admin"])
    args = parser.parse_args()
    {"generate-key": generate_key, "set-api-key": set_api_key, "init-admin": init_admin}[args.command]()


if __name__ == "__main__":
    main()
