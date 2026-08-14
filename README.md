# Maharashtra-focused Indian number-plate dataset

This project builds a reproducible one-class YOLO detection dataset (`number_plate`, class `0`) and a paired OCR crop collection. It is designed for Google Colab training and efficient inference on a Raspberry Pi 3B+ with Camera Module 3. The default build contains 12,000 full-scene images and can be expanded to 15,000.

## Dataset policy

- Seed: `20260814`; split: 80% train, 10% validation, 10% test.
- Validation and test are at least 80% real images, with every source family kept in one split.
- Synthetic positive plates are 60–70% Maharashtra (`MH`). Generated registrations are fictitious and checked against known real text.
- Hard negatives are 5–10%. Single/double-row plates, both viewpoints, broad vehicle groups, lighting/weather degradations, and private/commercial/EV/temporary styles are represented.
- The 2,122 reviewed crops from `../archive.zip` are preserved only as OCR samples, never as full-scene detector images.
- Only license-approved sources enter the build. Registry entries marked `verify` remain excluded unless their local decision file under `metadata/licenses/` contains `decision: allowed`.

Kaggle credentials stay in `D:\obj_det_dataset\kaggle.json` (or another directory selected with `KAGGLE_CONFIG_DIR`). Credential values are never copied into this project, manifests, logs, or reports.

## Setup and staged build

Run these commands from this project directory:

    python -m venv .venv
    .\.venv\Scripts\python -m pip install -e ".[dev]"
    $env:KAGGLE_CONFIG_DIR = 'D:\obj_det_dataset'
    .\.venv\Scripts\python scripts/download_sources.py --config config/default.yaml --dry-run
    .\.venv\Scripts\python scripts/download_sources.py --config config/default.yaml
    .\.venv\Scripts\python scripts/convert_annotations.py --config config/default.yaml
    .\.venv\Scripts\python scripts/generate_synthetic.py --config config/default.yaml --resume
    .\.venv\Scripts\python scripts/validate_dataset.py --config config/default.yaml

Generation writes temporary sibling files and atomically renames them. A rerun reuses an image-label pair only when both SHA-256 values match its manifest. `--no-resume` refuses a workspace with an existing manifest. Visual-QA rejects may be listed one output ID per line and supplied with `--reject-file`.

## Layout and label schema

    detection/images/{train,val,test}/*.jpg
    detection/labels/{train,val,test}/*.txt
    detection/data.yaml
    ocr/images/{train,val,test}/*
    ocr/labels.csv
    metadata/{source_manifest.csv,generation_manifest.csv,dataset_statistics.json}
    reports/{validation_report.json,contact_sheets/}
    notebooks/train_plate_detector_colab.ipynb

Each non-empty detector label contains `0 x_center y_center width height`, normalized to `[0,1]`. Intentional hard negatives have an empty matching `.txt`. The portable Ultralytics configuration is [detection/data.yaml](detection/data.yaml).

## Colab training

Upload the dataset as a ZIP or place it in Google Drive, open `notebooks/train_plate_detector_colab.ipynb`, set the dataset path, and run every cell. The workflow trains the nano detector at 512 pixels, validates once on the untouched test split, and exports ONNX and NCNN artifacts. Use a configurable batch size; automatic sizing is the safe default.

## Raspberry Pi 3B+ runtime flow

Keep the existing vehicle tracker as the first stage. Run the plate detector periodically inside each tracked vehicle crop, rank plate crops by size, sharpness, and confidence, and send only the best crops to OCR. Aggregate OCR predictions over several frames and accept a registration only after temporal consensus and Indian-format checks. This reduces CPU load substantially compared with full-frame OCR on every frame.

Synthetic data helps coverage, but final accuracy requires threshold tuning and evaluation on real Camera Module 3 footage from the installed pole height and distance. Collect both front and rear views plus daylight, night, rain, glare, occlusion, and motion cases.

## Validation gates

The validator requires matching readable image-label pairs, class `0`, finite in-bounds YOLO boxes, a minimum projected plate size of 8×4 pixels at 512 training resolution, checksum agreement, exact 80/10/10 counts, source-family isolation, at least 80% real validation/test data, 60–70% MH known positives, and 5–10% negatives. Contact-sheet captions intentionally omit plate text.

Generated bulk images, labels, raw downloads, OCR crops, and sensitive credentials are excluded from Git. Source code, configuration, notebooks, documentation, license decisions, and compact validation statistics remain versioned.
