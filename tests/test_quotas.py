from dataclasses import replace
from pathlib import Path

from plate_dataset import quotas as quota_module
from plate_dataset.builder import _synthetic_specs
from plate_dataset.config import load_config
from plate_dataset.quotas import generation_quotas
from plate_dataset.records import Box, ImageRecord


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
    assert {name: quota.double_row_positives for name, quota in quotas.items()} == {
        "train": 7_400,
        "val": 925,
        "test": 925,
    }
    assert {name: quota.low_light for name, quota in quotas.items()} == {
        "train": 10_000,
        "val": 1_250,
        "test": 1_250,
    }
    assert {name: quota.adverse for name, quota in quotas.items()} == {
        "train": 6_000,
        "val": 750,
        "test": 750,
    }
    assert {name: quota.distance for name, quota in quotas.items()} == {
        "train": 3_000,
        "val": 375,
        "test": 375,
    }
    assert {name: quota.occlusion for name, quota in quotas.items()} == {
        "train": 3_000,
        "val": 375,
        "test": 375,
    }


def test_slot_plan_is_seeded_distributed_and_not_prefix_coupled() -> None:
    """Catches quota dimensions sharing prefix slots or the same ranking salt."""
    config = load_config(Path("config/default.yaml"))
    quotas = generation_quotas(config)

    plans = {
        split: quota_module.generation_slot_plan(config, split)
        for split in ("train", "val", "test")
    }

    assert plans == {
        split: quota_module.generation_slot_plan(config, split)
        for split in ("train", "val", "test")
    }
    assert plans != {
        split: quota_module.generation_slot_plan(replace(config, seed=config.seed + 1), split)
        for split in ("train", "val", "test")
    }
    for split, plan in plans.items():
        quota = quotas[split]
        positives = frozenset(range(quota.total)) - plan.negative_slots
        effects = {
            name: frozenset(
                slot for slot, condition in enumerate(plan.conditions) if condition == name
            )
            for name in ("night", "rain", "distance", "occlusion")
        }

        assert len(plan.negative_slots) == quota.negatives
        assert len(plan.mh_positive_slots) == quota.mh_positives
        assert len(plan.double_row_positive_slots) == quota.double_row_positives
        assert plan.mh_positive_slots <= positives
        assert plan.double_row_positive_slots <= positives
        assert effects["occlusion"] <= positives
        assert {name: len(slots) for name, slots in effects.items()} == {
            "night": quota.low_light,
            "rain": quota.adverse,
            "distance": quota.distance,
            "occlusion": quota.occlusion,
        }

        assert plan.negative_slots != frozenset(range(quota.negatives))
        assert plan.mh_positive_slots != frozenset(sorted(positives)[: quota.mh_positives])
        assert plan.double_row_positive_slots != frozenset(
            sorted(positives)[: quota.double_row_positives]
        )
        assert not plan.negative_slots <= effects["night"]
        assert not plan.double_row_positive_slots <= plan.mh_positive_slots
        for selected in (
            plan.negative_slots,
            plan.mh_positive_slots,
            plan.double_row_positive_slots,
            *effects.values(),
        ):
            assert all(
                selected & frozenset(range(start, min(start + quota.total // 4, quota.total)))
                for start in range(0, quota.total, quota.total // 4)
            )


def test_synthetic_specs_apply_the_independent_slot_plan() -> None:
    """Catches the builder reverting to prefix quota assignment instead of the slot plan."""
    config = load_config(Path("config/default.yaml"))
    records = [
        ImageRecord(
            record_id=f"record-{index}",
            image_path=Path(f"unused-{index}.jpg"),
            width=320,
            height=180,
            boxes=(Box(0, 100, 120, 220, 155),),
            source_id="fixture",
            source_family=f"family-{index}",
            is_real=True,
            plate_text=None,
            tags={"vehicle_type": "car", "viewpoint": "rear"},
        )
        for index in range(3)
    ]

    specs = _synthetic_specs(config, records, frozenset())

    for split in ("train", "val", "test"):
        chosen = [spec for spec in specs if spec.split == split]
        plan = quota_module.generation_slot_plan(config, split)
        assert frozenset(slot for slot, spec in enumerate(chosen) if spec.negative) == (
            plan.negative_slots
        )
        assert frozenset(slot for slot, spec in enumerate(chosen) if spec.force_mh) == (
            plan.mh_positive_slots
        )
        assert frozenset(
            slot for slot, spec in enumerate(chosen) if spec.layout == "double"
        ) == plan.double_row_positive_slots
        assert tuple(spec.condition for spec in chosen) == plan.conditions
