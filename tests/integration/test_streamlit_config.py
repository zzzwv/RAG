import tomllib
from pathlib import Path


def test_streamlit_disables_module_file_watcher():
    config_path = Path(".streamlit/config.toml")
    assert config_path.exists()
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert config["server"]["fileWatcherType"] == "none"
    assert config["server"]["runOnSave"] is False
