# 50,000-image Indian number-plate synthetic dataset

This project builds a reproducible one-class YOLO detection dataset (`number_plate`, class `0`) and a paired OCR crop collection. It creates the dataset only; detector training, model export, and real-camera benchmarking remain separate tasks.

## Completed build: 15 August 2026

The requested dataset has been generated in this exact folder and fully validated.

| Item | Completed count / result |
| --- | --- |
| Detector images | 50,000 |
| YOLO label files | 50,000 |
| Training pairs | 40,000 |
| Validation pairs | 5,000 |
| Test pairs | 5,000 |
| Positive plate scenes | 46,250 |
| Empty-label hard negatives | 3,750 |
| New synthetic OCR crops | 46,250 |
| Preserved original OCR crops | 2,122 |
| OCR label rows in total | 48,372 |
| Normalized licensed source records | 2,181 |
| Detector/OCR artifact footprint | about 1.72 GiB |
| Full validation | 50,000 images + 50,000 labels, 0 errors |

The validator emits two non-blocking provenance warnings because the approved source packages do not supply verified vehicle type or viewpoint metadata. Those fields remain `unknown` instead of being guessed; this does not affect YOLO labels, detector training, OCR crops, or OCR labels.

See [README/DATASET_INVENTORY.md](README/DATASET_INVENTORY.md) for a complete folder inventory, exact distribution counts, validation details, and the Colab handoff checklist.

## Fixed output profile

- Seed: `20260814`.
- Detector pairs: exactly 50,000 images and 50,000 matching labels.
- Split: 40,000 train, 5,000 validation, and 5,000 test pairs.
- Positives: 46,250 scenes and at least 46,250 linked synthetic OCR crops.
- Hard negatives: 3,750 empty-label scenes: 3,000 train, 375 validation, and 375 test.
- Maharashtra registrations: exactly 30,063 (65% after deterministic rounding) across positive scenes; this remains within the approved 60–70% range.
- Double-row plates: exactly 9,250 (20%) across positive scenes; this remains within the approved 15–30% range.
- Full scenes: longest edge at most 960 pixels, JPEG quality 88.
- OCR crops: aspect-preserving 256×128 padded JPEGs.
- Existing OCR corpus: preserve the separately identified 2,122 reviewed crops from `../archive.zip`; they do not count toward the synthetic minimum.

Every visible final plate is newly rendered and fictitious. Real source scenes supply geometry and background context, but unmodified source images do not enter the detector outputs.

## Source and credential policy

Two registry sources are allowed by default and recorded as CC0-1.0:

1. `kedarsai/indian-license-plates-with-labels`
2. `deepakat002/indian-vehicle-number-plate-yolo-annotation`

Sources marked `verify` or `blocked` are excluded. A `verify` source can enter normalization only after a local `metadata/licenses/<source>.yaml` file records `decision: allowed`; its license must be reviewed independently first. Conversion ignores archives without an allowed decision and atomically replaces `metadata/normalized_records.jsonl` only after every selected archive has normalized successfully.

Keep Kaggle credentials outside this repository. The commands below use `D:\obj_det_dataset\kaggle.json` through `KAGGLE_CONFIG_DIR`; credential contents are never copied into logs, manifests, or reports.

## Setup and staged build

Run from this project directory in PowerShell:

    python -m venv .venv
    .\.venv\Scripts\python -m pip install -e ".[dev]"
    $env:KAGGLE_CONFIG_DIR = 'D:\obj_det_dataset'
    .\.venv\Scripts\python scripts/download_sources.py --config config/default.yaml --dry-run
    .\.venv\Scripts\python scripts/download_sources.py --config config/default.yaml
    .\.venv\Scripts\python scripts/convert_annotations.py --config config/default.yaml
    .\.venv\Scripts\python scripts/generate_synthetic.py --config config/default.yaml --dry-run --workers 0
    .\.venv\Scripts\python scripts/generate_synthetic.py --config config/default.yaml --resume --workers 0
    .\.venv\Scripts\python scripts/validate_dataset.py --config config/default.yaml

The generation dry-run reads normalized source images, prints the exact split quotas, projected byte use, required 5 GB reserve, and current free bytes, and creates no detector files. Storage depends on measured source JPEG density. The design target is a final footprint below 25 GB; generation refuses to start unless the projected build can still leave at least 5 GB free. Do not start a second generator against the same workspace.

`--workers 0` selects a bounded automatic worker count; use `--workers N` to set it explicitly. Generation prints progress after each 500 completed outputs and checkpoints the manifest during the build.

## Resume and visual-QA replacement

Generation writes each image, detector label, and OCR crop through a temporary sibling followed by atomic rename. `--resume` is the default: a prior output is reused only when every required file exists and its SHA-256 matches the manifest. `--no-resume` refuses to continue when a generation manifest already exists, protecting against accidental overwrite.

To replace images rejected during visual review, put one generation `output_id` per line in a text file and rerun:

    .\.venv\Scripts\python scripts/generate_synthetic.py --config config/default.yaml --resume --workers 0 --reject-file rejected_ids.txt

Rejected IDs are omitted and replaced by later deterministic variants without deleting unrelated files. After generation, `ocr/labels.csv` is rebuilt from preserved existing OCR labels plus the current synthetic manifest, so removed synthetic IDs do not remain in the label table.

## Output layout

    detection/images/{train,val,test}/*.jpg
    detection/labels/{train,val,test}/*.txt
    detection/data.yaml
    ocr/images/{train,val,test}/*.jpg
    ocr/labels.csv
    metadata/licenses/*
    metadata/source_manifest.csv
    metadata/normalized_records.jsonl
    metadata/generation_manifest.csv
    metadata/dataset_statistics.json
    reports/validation_report.json
    reports/contact_sheets/*.jpg

Each non-empty detector label contains `0 x_center y_center width height`, normalized to `[0,1]`. Intentional hard negatives have an empty matching `.txt`. The portable Ultralytics configuration is [detection/data.yaml](detection/data.yaml).

## Validation and the synthetic-holdout limitation

The validator requires exact image, label, split, positive, negative, and OCR-crop counts; readable files; class-`0` finite in-bounds YOLO boxes; minimum projected plate size; checksum agreement; source-family isolation; and the specified registration, layout, lighting, and adverse-condition distributions. It exits nonzero whenever the report contains an error.

Validation and test use source families disjoint from training, but their visible plates are still synthetic. They measure pipeline consistency and held-out source-scene performance, not real-world Camera Module 3 accuracy. A clean validation report must not be presented as a real-camera benchmark.

## Colab handoff

After local validation succeeds, package or upload the generated `detection/` directory and its `data.yaml` to Google Drive. In Colab, copy or mount the data into the runtime, extract it if uploaded as an archive, and confirm that `data.yaml` resolves `images/train`, `images/val`, and `images/test` before starting training. Install the training framework in Colab rather than adding it to this dataset-builder environment. Keep the test split untouched until the final model choice is fixed.

The 50,000-image detector package can be large, so Google Drive transfer is more reliable than browser upload. Preserve the directory structure and compare archive or file checksums after transfer when possible.

## Camera Module 3 evaluation caveat

Final deployment thresholds and OCR acceptance rules require footage from the actual Raspberry Pi 3B+ and Camera Module 3 installation. Evaluate at the installed pole height, angle, and distance with front and rear vehicles across daylight, night, rain, glare, occlusion, and motion. Keep the vehicle tracker as the first stage, run plate detection within tracked vehicle crops, rank candidate crops, and aggregate OCR over multiple frames. Synthetic holdout metrics cannot replace that evaluation.

## GitHub dataset access

The published repository stores the generated JPEG images and source archives with Git LFS, so the normal Git history stays manageable while the full dataset remains available after cloning. Install Git LFS before cloning, then run `git lfs pull` if the images were not downloaded automatically. Source code, labels, metadata, reports, configuration, and documentation are versioned normally. Kaggle credentials, virtual environments, caches, and working trees are never included.
