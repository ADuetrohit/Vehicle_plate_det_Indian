from pathlib import Path

from plate_dataset.config import load_config
from plate_dataset.quotas import generation_quotas


def test_50000_quota_is_exact() -> None:
    """Catches generated split quotas that miss an approved sample count."""
    quotas = generation_quotas(load_config(Path("config/default.yaml")))

    assert {name: quota.total for name, quota in quotas.items()} == {
        "train": 40_000,
        "val": 5_000,
        "test": 5_000,
    }
    assert {name: quota.negatives for name, quota in quotas.items()} == {
        "train": 3_000,
        "val": 375,
        "test": 375,
    }
    assert sum(quota.positives for quota in quotas.values()) == 46_250
    assert 0.60 <= sum(quota.mh_positives for quota in quotas.values()) / 46_250 <= 0.70
