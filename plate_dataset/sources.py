from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SourceSpec:
    slug: str
    expected_license: str
    license_status: Literal["allowed", "verify", "blocked"]
    annotation_format: Literal["yolo", "voc"]
    priority: int


def source_registry() -> tuple[SourceSpec, ...]:
    return (
        SourceSpec(
            "kedarsai/indian-license-plates-with-labels",
            "CC0-1.0",
            "allowed",
            "yolo",
            10,
        ),
        SourceSpec(
            "deepakat002/indian-vehicle-number-plate-yolo-annotation",
            "CC0-1.0",
            "allowed",
            "yolo",
            20,
        ),
        SourceSpec(
            "saisirishan/indian-vehicle-dataset",
            "unknown",
            "verify",
            "voc",
            30,
        ),
        SourceSpec(
            "gauravsanwal/indian-licence-plate",
            "unknown",
            "verify",
            "yolo",
            40,
        ),
    )

