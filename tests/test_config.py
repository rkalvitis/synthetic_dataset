import os

from popsynth.config import load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_mac_profile_inherits_defaults():
    base = load_config(os.path.join(ROOT, "configs", "default.yaml"))
    mac = load_config(os.path.join(ROOT, "configs", "mac_preview.yaml"), ["compose.num_images=3"])
    assert mac["render"]["size"] == 1024 and mac["render"]["supersample"] == 1
    assert mac["render"]["camera"] == base["render"]["camera"]       # untouched sub-dict inherited
    assert mac["compose"]["num_images"] == 3 and mac["compose"]["object_size"] == base["compose"]["object_size"]
    assert "_base" not in mac
