from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PIL import Image
import pytest

from plate_dataset.adapters.voc import parse_voc
from plate_dataset.adapters.yolo import parse_yolo
from plate_dataset.dedupe import perceptual_family
from plate_dataset.ingest import ingest_source
from plate_dataset.records import Box, ImageRecord
from plate_dataset.sources import SourceSpec


def _image(path: Path, size: tuple[int, int] = (200, 100), color: str = "gray") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)
    return path


def _voc_xml(path: Path, image_name: str, object_name: str = "license_plate") -> Path:
    path.write_text(
        f"""<annotation>
<filename>{image_name}</filename>
<size><width>200</width><height>100</height><depth>3</depth></size>
<object><name>{object_name}</name><bndbox>
<xmin>20</xmin><ymin>30</ymin><xmax>120</xmax><ymax>60</ymax>
</bndbox></object>
</annotation>""",
        encoding="utf-8",
    )
    return path


def test_parse_voc_reads_plate_box_from_real_image_dimensions(tmp_path: Path) -> None:
    """Catches VOC axis swaps and trust in stale XML dimensions."""
    image = _image(tmp_path / "a.jpg")
    xml = _voc_xml(tmp_path / "a.xml", image.name)

    record = parse_voc(xml, image, "voc-source")

    assert (record.width, record.height) == (200, 100)
    assert record.boxes == (Box(0, 20, 30, 120, 60),)
    assert record.source_id == "voc-source"


def test_parse_voc_ignores_non_plate_objects(tmp_path: Path) -> None:
    """Catches vehicle or person boxes leaking into the one-class dataset."""
    image = _image(tmp_path / "a.jpg")
    xml = _voc_xml(tmp_path / "a.xml", image.name, object_name="car")

    with pytest.raises(ValueError, match="no number-plate boxes"):
        parse_voc(xml, image, "voc-source")


def test_parse_yolo_denormalizes_plate_box(tmp_path: Path) -> None:
    """Catches incorrect conversion from normalized center-size coordinates."""
    image = _image(tmp_path / "a.jpg")
    label = tmp_path / "a.txt"
    label.write_text("0 0.35 0.45 0.5 0.3\n", encoding="utf-8")

    record = parse_yolo(label, image, "yolo-source")

    assert record.boxes == (Box(0, 20, 30, 120, 60),)


@pytest.mark.parametrize(
    "row",
    ["1 0.5 0.5 0.2 0.2", "0 nan 0.5 0.2 0.2", "0 0.5 0.5 -0.2 0.2"],
)
def test_parse_yolo_rejects_invalid_rows(tmp_path: Path, row: str) -> None:
    """Catches extra classes, non-finite coordinates, and negative sizes."""
    image = _image(tmp_path / "a.jpg")
    label = tmp_path / "a.txt"
    label.write_text(row + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid YOLO label"):
        parse_yolo(label, image, "yolo-source")


def test_empty_yolo_label_requires_explicit_negative_flag(tmp_path: Path) -> None:
    """Catches missing annotations being silently converted into negatives."""
    image = _image(tmp_path / "a.jpg")
    label = tmp_path / "a.txt"
    label.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty label"):
        parse_yolo(label, image, "yolo-source")

    negative = parse_yolo(label, image, "yolo-source", allow_negative=True)
    assert negative.boxes == ()
    assert negative.tags["negative"] == "true"


def test_identical_images_receive_same_source_family(tmp_path: Path) -> None:
    """Catches duplicate scenes leaking across dataset splits."""
    first_path = _image(tmp_path / "first.jpg", color="navy")
    second_path = _image(tmp_path / "second.jpg", color="navy")
    base = ImageRecord(
        record_id="first",
        image_path=first_path,
        width=200,
        height=100,
        boxes=(Box(0, 20, 30, 120, 60),),
        source_id="source",
        source_family="first",
        is_real=True,
        plate_text=None,
        tags={},
    )
    records = [base, replace(base, record_id="second", image_path=second_path)]

    mapping = perceptual_family(records, max_distance=4)

    assert mapping["first"] == mapping["second"] == "first"


def test_ingest_source_discovers_matching_yolo_pairs(tmp_path: Path) -> None:
    """Catches source layouts being scanned without matching image-label pairs."""
    root = tmp_path / "source"
    image = _image(root / "images" / "a.jpg")
    label = root / "labels" / "a.txt"
    label.parent.mkdir(parents=True)
    label.write_text("0 0.35 0.45 0.5 0.3\n", encoding="utf-8")
    spec = SourceSpec("sample/source", "CC0-1.0", "allowed", "yolo", 1)

    records = ingest_source(root, spec)

    assert len(records) == 1
    assert records[0].image_path == image
    assert records[0].boxes == (Box(0, 20, 30, 120, 60),)
