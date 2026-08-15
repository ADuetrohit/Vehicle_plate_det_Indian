# Generated Dataset Inventory

## What was created

This folder contains a finished synthetic Indian number-plate dataset for one-class YOLO plate detection and a companion OCR corpus. All visible final registration numbers are fictitious synthetic registrations. The licensed source images were used only as scene/geometry anchors; source registrations were erased before final detector images were written.

Build date: 15 August 2026  
Dataset location: `D:\obj_det_dataset\synthetic_number_plate_dataset`

## Exact completed counts

| Asset | Train | Validation | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| Detector JPEG images | 40,000 | 5,000 | 5,000 | 50,000 |
| YOLO `.txt` label files | 40,000 | 5,000 | 5,000 | 50,000 |
| Positive visible-plate scenes | 37,000 | 4,625 | 4,625 | 46,250 |
| Intentional empty-label negatives | 3,000 | 375 | 375 | 3,750 |
| Synthetic OCR crop images | 37,000 | 4,625 | 4,625 | 46,250 |

The OCR collection also retains 2,122 separately imported original crops. Therefore `ocr/labels.csv` contains 48,372 label rows in total:

- 46,250 synthetic crop/text rows;
- 2,122 preserved original crop/text rows.

## Folder-by-folder contents

| Path | Contents |
| --- | --- |
| `detection/images/train` | 40,000 JPEG detector training images |
| `detection/images/val` | 5,000 JPEG validation images |
| `detection/images/test` | 5,000 JPEG test images |
| `detection/labels/{train,val,test}` | One matching YOLO class-`0` label file per detector image; hard negatives are intentionally empty |
| `detection/data.yaml` | Portable one-class Ultralytics/YOLO dataset configuration (`number_plate`) |
| `ocr/images/{train,val,test}` | Synthetic OCR crops plus the preserved OCR crop import |
| `ocr/labels.csv` | OCR crop path, exact registration text, split, origin, and reconciliation metadata |
| `metadata/generation_manifest.csv` | One row per generated detector output, including split, source family, fictitious plate text, labels/crops, hashes, condition, and eligibility |
| `metadata/normalized_records.jsonl` | 2,181 normalized licensed source records used by the builder |
| `metadata/source_manifest.csv` | Downloaded allowed-source archive records and checksums |
| `metadata/rejected_source_records.csv` | Source records excluded during safe normalization, with reasons |
| `metadata/dataset_statistics.json` | Machine-readable final totals and distributions |
| `reports/validation_report.json` | Full validation result: zero errors and two metadata-only warnings |
| `reports/contact_sheets` | Contact sheets for visual sampling across splits, plate styles, layouts, effects, and negatives |
| `raw` | Downloaded/extracted CC0 source material used only to reproduce the build |

## Detector label format

This is a one-class YOLO dataset:

```text
0 x_center y_center width height
```

All values are normalized to the image width/height. Class `0` means `number_plate`. The matching configuration is `detection/data.yaml`.

## Dataset composition

### Registration and layout

- Maharashtra (`MH`) positives: 30,063 of 46,250 positives (65.0%).
- Other Indian-state positives: 16,187.
- Single-row plates: 37,000.
- Double-row plates: 9,250 (20.0% of positives).
- Hard negatives: 3,750 with an empty label file and no OCR crop.

### Plate-style distribution

| Plate style | Count |
| --- | ---: |
| Private white | 9,254 |
| Commercial yellow | 9,299 |
| Electric private green | 9,246 |
| Electric commercial green | 9,235 |
| Temporary | 9,216 |
| Removed/negative | 3,750 |

### Condition distribution

| Condition | Count |
| --- | ---: |
| Day | 3,750 |
| Night / low light | 12,500 |
| Rain | 7,500 |
| Fog | 3,750 |
| Glare | 3,750 |
| Shadow | 3,750 |
| Motion | 3,750 |
| Compression | 3,750 |
| Distance | 3,750 |
| Occlusion | 3,750 |

## Source and safety record

Only these CC0-approved Kaggle source packages were used:

1. `kedarsai/indian-license-plates-with-labels`
2. `deepakat002/indian-vehicle-number-plate-yolo-annotation`

The two registry entries with unverified terms were excluded. Kaggle credentials were read only from `KAGGLE_CONFIG_DIR`; no credential value is stored in this project, dataset manifest, report, or README.

## Validation result

The completed run checked all 50,000 images and 50,000 labels and reported:

```text
images=50000 labels=50000 errors=0 warnings=2
```

Validation confirmed exact split counts, hard-negative counts, valid class-`0` YOLO geometry, readable files, detector-size eligibility, image/label/crop checksum agreement, split isolation by source family, Maharashtra/layout/condition quotas, and OCR linkage.

The two warnings are deliberate metadata disclosures: the imported source packages did not provide verified vehicle type or viewpoint, so both fields are recorded as `unknown` rather than invented. They do not affect detector images, YOLO labels, OCR crops, OCR text, or Colab training.

## Colab handoff checklist

1. Upload or copy the entire `detection` folder to Google Drive or the Colab runtime. Keep `images`, `labels`, and `data.yaml` together.
2. Confirm `detection/data.yaml` resolves these relative paths:

   ```text
   train: images/train
   val: images/val
   test: images/test
   names:
     0: number_plate
   ```

3. Train only from the `train` split; use `val` for model selection.
4. Keep `test` untouched until the final model is selected.
5. Use `ocr/labels.csv` and the OCR crop images only for the separate OCR model/evaluation work.
6. After Colab training, test the detector against actual Raspberry Pi Camera Module 3 footage from the final pole/road installation. Synthetic validation is a consistency check, not a replacement for live camera evaluation.

## Reproduction commands

Run these from the project root only when you intentionally want to reproduce or resume the dataset build:

```powershell
$env:KAGGLE_CONFIG_DIR = 'D:\obj_det_dataset'
.\.venv\Scripts\python scripts\download_sources.py --config config/default.yaml
.\.venv\Scripts\python scripts\convert_annotations.py --config config/default.yaml
.\.venv\Scripts\python scripts\generate_synthetic.py --config config/default.yaml --resume --workers 0
.\.venv\Scripts\python scripts\validate_dataset.py --config config/default.yaml
```

The generator is resumable and checks hashes before reusing an existing output. Do not run two generators at the same time against this folder.
