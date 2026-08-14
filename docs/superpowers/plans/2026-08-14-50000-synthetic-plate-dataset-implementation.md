# 50,000-Image Synthetic Plate Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce and validate exactly 50,000 synthetic full-scene Indian number-plate detector samples and at least 46,250 linked synthetic OCR crops inside the existing project.

**Architecture:** Extend the tested pipeline with a synthetic-only build profile, deterministic per-split quotas, family-safe source pools, OCR crop export, storage preflight, and checkpointed parallel rendering. License-approved real images provide background geometry, but every final visible plate is rendered synthetically or removed for a hard negative.

**Tech Stack:** Python 3.10, Pillow, NumPy, OpenCV headless, PyYAML, ImageHash, Kaggle CLI, concurrent futures, pytest, and Ultralytics-compatible YOLO text labels.

## Global Constraints

- Write only under `D:\obj_det_dataset\synthetic_number_plate_dataset`.
- Preserve all existing archives, credentials, models, CSV files, and 2,122 imported OCR crops.
- Use seed `20260814`.
- Generate exactly 50,000 detector images and 50,000 matching label files.
- Use exact split counts: 40,000 train, 5,000 validation, and 5,000 test.
- Generate exactly 3,750 hard negatives: 3,000 train, 375 validation, and 375 test.
- Generate at least 46,250 linked synthetic OCR crops.
- Keep Maharashtra positive plates between 60% and 70%, targeting 65%.
- Keep double-row positives between 15% and 30%, targeting 20%.
- Keep low-light scenes between 20% and 30% and adverse conditions at or above 15%.
- Keep source families and their variants in one split.
- Include only sources with an allowed recorded license decision.
- Use one detector class: `number_plate`, class ID `0`.
- Limit full-scene longest edge to 960 pixels and encode JPEG at quality 88.
- Encode OCR crops on a 256×128 aspect-preserving padded canvas.
- Refuse generation when projected output would leave less than 5 GB free.
- Bulk data remains ignored by Git.
- Full detector training is outside this plan.

## File Map

- Modify `config/default.yaml` and `plate_dataset/config.py`: expose the synthetic-only scale and output profile.
- Create `plate_dataset/quotas.py`: calculate exact per-split positive, negative, MH, layout, and condition quotas.
- Modify `plate_dataset/split.py`: add source-family allocation for synthetic-only output.
- Create `plate_dataset/storage.py`: estimate output footprint and enforce disk headroom.
- Create `plate_dataset/ocr_export.py`: crop, pad, checksum, and merge synthetic OCR labels.
- Refactor `plate_dataset/builder.py`: prepare deterministic work specifications, render outputs, checkpoint manifests, resume, reject, and parallelize.
- Modify `plate_dataset/manifests.py`: add OCR path/checksum fields and checkpoint-safe writes.
- Modify `plate_dataset/validate.py` and `plate_dataset/reporting.py`: enforce the approved synthetic-only acceptance gates and generate representative QA sheets.
- Modify `scripts/generate_synthetic.py`, `scripts/convert_annotations.py`, `scripts/validate_dataset.py`, and `README.md`: expose the full run.
- Modify tests under `tests/`: cover each behavior before implementation.

---

### Task 1: Synthetic-only configuration and exact quotas

**Files:**
- Modify: `plate_dataset/config.py`
- Modify: `config/default.yaml`
- Create: `plate_dataset/quotas.py`
- Modify: `tests/test_config_records.py`
- Create: `tests/test_quotas.py`

**Interfaces:**
- Consumes: existing `BuildConfig` and `split_target_counts`.
- Produces: `GenerationQuota` and `generation_quotas(config: BuildConfig) -> dict[str, GenerationQuota]`.

- [ ] **Step 1: Write failing configuration tests**

Add:

    def test_default_config_is_approved_50000_synthetic_build():
        cfg = load_config(Path("config/default.yaml"))
        assert cfg.target_images == cfg.max_images == 50_000
        assert cfg.synthetic_only is True
        assert cfg.negative_share == (0.075, 0.075)
        assert cfg.max_scene_edge == 960
        assert cfg.jpeg_quality == 88
        assert cfg.ocr_canvas == (256, 128)
        assert cfg.min_free_gb == 5.0

- [ ] **Step 2: Write failing quota tests**

Create `tests/test_quotas.py`:

    def test_50000_quota_is_exact():
        quotas = generation_quotas(load_config(Path("config/default.yaml")))
        assert {name: q.total for name, q in quotas.items()} == {
            "train": 40_000, "val": 5_000, "test": 5_000,
        }
        assert {name: q.negatives for name, q in quotas.items()} == {
            "train": 3_000, "val": 375, "test": 375,
        }
        assert sum(q.positives for q in quotas.values()) == 46_250
        assert 0.60 <= sum(q.mh_positives for q in quotas.values()) / 46_250 <= 0.70

- [ ] **Step 3: Run the tests and verify RED**

Run:

    .\.venv\Scripts\python -m pytest tests/test_config_records.py tests/test_quotas.py -v

Expected: failure because the output-profile fields and quota module do not exist.

- [ ] **Step 4: Implement the configuration fields and quota dataclass**

Append defaulted fields to `BuildConfig` so existing small fixtures remain compatible:

    synthetic_only: bool = False
    max_scene_edge: int = 960
    jpeg_quality: int = 88
    ocr_canvas: tuple[int, int] = (256, 128)
    min_free_gb: float = 5.0
    workers: int = 0

Define:

    @dataclass(frozen=True)
    class GenerationQuota:
        split: Literal["train", "val", "test"]
        total: int
        positives: int
        negatives: int
        mh_positives: int
        double_row_positives: int
        low_light: int
        adverse: int

Use largest-remainder allocation so every field sums exactly to its global total while retaining the 80/10/10 proportions.

- [ ] **Step 5: Run GREEN and commit**

Run:

    .\.venv\Scripts\python -m pytest tests/test_config_records.py tests/test_quotas.py -v
    git diff --check

Commit:

    git add config/default.yaml plate_dataset/config.py plate_dataset/quotas.py tests/test_config_records.py tests/test_quotas.py
    git commit -m "feat: configure exact 50000-image generation quotas"

### Task 2: Synthetic source-family allocation

**Files:**
- Modify: `plate_dataset/split.py`
- Modify: `tests/test_split_builder.py`

**Interfaces:**
- Consumes: `ImageRecord`, `BuildConfig`, and `SplitAssignment`.
- Produces: synthetic-only behavior through the existing `assign_splits(records, config)` signature.

- [ ] **Step 1: Write the failing family-pool test**

Add:

    def test_synthetic_only_assigns_all_source_families_without_real_holdout(tmp_path):
        config = replace(_config(tmp_path, target=50), synthetic_only=True)
        records = [_record(i, is_real=True) for i in range(20)]
        assignments = assign_splits(records, config)
        assert set(assignments) == {record.record_id for record in records}
        assert {item.split for item in assignments.values()} == {"train", "val", "test"}
        by_family = defaultdict(set)
        for record in records:
            by_family[record.source_family].add(assignments[record.record_id].split)
        assert all(len(values) == 1 for values in by_family.values())

- [ ] **Step 2: Run RED**

Run:

    .\.venv\Scripts\python -m pytest tests/test_split_builder.py::test_synthetic_only_assigns_all_source_families_without_real_holdout -v

Expected: failure from the existing real-holdout calculation.

- [ ] **Step 3: Implement deterministic family-pool allocation**

When `config.synthetic_only` is true, hash `seed:source_family`, sort whole families, and greedily allocate record counts toward 80/10/10 source-pool targets. Require at least one family in each split. Preserve the existing hybrid/real-holdout branch unchanged when `synthetic_only` is false.

- [ ] **Step 4: Run GREEN and commit**

Run:

    .\.venv\Scripts\python -m pytest tests/test_split_builder.py -v

Commit:

    git add plate_dataset/split.py tests/test_split_builder.py
    git commit -m "feat: allocate synthetic source families across splits"

### Task 3: Storage preflight and OCR crop export

**Files:**
- Create: `plate_dataset/storage.py`
- Create: `plate_dataset/ocr_export.py`
- Modify: `plate_dataset/manifests.py`
- Create: `tests/test_storage_ocr_export.py`

**Interfaces:**
- Produces: `StorageEstimate`, `estimate_storage(config, source_samples) -> StorageEstimate`, `require_storage(estimate, free_bytes) -> None`, `export_ocr_crop(image, box, output_path, canvas, quality) -> str`, and `merge_ocr_labels(existing_csv, synthetic_rows, output_csv) -> int`.

- [ ] **Step 1: Write failing storage tests**

    def test_storage_preflight_rejects_less_than_required_headroom():
        estimate = StorageEstimate(projected_bytes=10_000, reserve_bytes=5_000)
        with pytest.raises(InsufficientStorage, match="required_bytes=15000"):
            require_storage(estimate, free_bytes=14_999)

    def test_storage_estimate_uses_measured_source_average(sample_jpegs, config):
        result = estimate_storage(config, sample_jpegs)
        assert result.projected_bytes > sum(path.stat().st_size for path in sample_jpegs)

- [ ] **Step 2: Write failing OCR crop tests**

    def test_exported_crop_is_256_by_128_and_checksum_valid(scene, tmp_path):
        output = tmp_path / "crop.jpg"
        checksum = export_ocr_crop(
            scene, Box(0, 20, 20, 180, 70), output, (256, 128), 92
        )
        assert Image.open(output).size == (256, 128)
        assert checksum == sha256_file(output)

    def test_merge_preserves_existing_and_adds_synthetic(existing_labels, tmp_path):
        count = merge_ocr_labels(existing_labels, [synthetic_row], tmp_path / "labels.csv")
        assert count == 3

- [ ] **Step 3: Run RED**

Run:

    .\.venv\Scripts\python -m pytest tests/test_storage_ocr_export.py -v

Expected: import failures for both new modules.

- [ ] **Step 4: Implement storage and crop modules**

Measure source JPEG bytes per pixel, apply the 960-edge scale, multiply by 50,000, add 256×128 crop estimates, raw-source size, a 15% temporary margin, and the 5 GB reserve. Crop the final augmented RGB scene with 10% padding, resize without distortion, center it on a 256×128 neutral canvas, save atomically, and return SHA-256.

Extend manifest fields with `ocr_path` and `ocr_sha256`. Merge OCR CSV rows by stable `source_id:image_name`, preserving all existing rows and writing synthetic rows sorted by split and output ID.

- [ ] **Step 5: Run GREEN and commit**

Run:

    .\.venv\Scripts\python -m pytest tests/test_storage_ocr_export.py -v

Commit:

    git add plate_dataset/storage.py plate_dataset/ocr_export.py plate_dataset/manifests.py tests/test_storage_ocr_export.py
    git commit -m "feat: preflight storage and export linked OCR crops"

### Task 4: Deterministic synthetic-only rendering and resume

**Files:**
- Modify: `plate_dataset/builder.py`
- Modify: `tests/test_split_builder.py`

**Interfaces:**
- Consumes: generation quotas, source assignments, output profile, crop exporter, and prior manifest.
- Produces: `GenerationSpec`, `GenerationResult`, and the extended `build_dataset(config, records, output, rejected_ids=frozenset(), progress=None) -> BuildManifest`.

- [ ] **Step 1: Write failing synthetic-only integration test**

Add a 20-output fixture with four source families:

    def test_synthetic_only_builder_writes_only_synthetic_outputs_and_crops(tmp_path):
        config = replace(
            _config(tmp_path / "dataset", target=20),
            synthetic_only=True,
            negative_share=(0.10, 0.10),
            max_scene_edge=160,
            ocr_canvas=(256, 128),
            min_free_gb=0,
        )
        result = build_dataset(config, _write_real_records(tmp_path, 8), config.workspace)
        rows = list(csv.DictReader(result.manifest_path.open(newline="", encoding="utf-8")))
        assert len(rows) == 20
        assert {row["origin"] for row in rows} == {"synthetic"}
        assert sum(row["negative"] == "true" for row in rows) == 2
        assert sum(bool(row["ocr_path"]) for row in rows) == 18
        assert all(max(Image.open(config.workspace / row["image_path"]).size) <= 160 for row in rows)

- [ ] **Step 2: Write failing resume and rejection tests**

    def test_synthetic_only_resume_checks_image_label_and_crop_checksums(...):
        first = build_dataset(config, records, output)
        second = build_dataset(config, records, output)
        assert second.reused_count == 20

    def test_rejected_output_is_replaced_without_changing_total(...):
        first = build_dataset(config, records, output)
        rejected = first_rows[0]["output_id"]
        second = build_dataset(config, records, output, rejected_ids={rejected})
        second_ids = {row["output_id"] for row in read_manifest(second.manifest_path)}
        assert rejected not in second_ids
        assert len(second_ids) == 20

- [ ] **Step 3: Run RED**

Run:

    .\.venv\Scripts\python -m pytest tests/test_split_builder.py -v

Expected: the current builder copies source images and has no OCR/rejection interface.

- [ ] **Step 4: Implement deterministic work specifications**

Create one `GenerationSpec` per quota slot with split, source record, global index, variant index, negative flag, MH flag, layout, category, and condition. Cycle source families evenly inside their assigned split. Derive output IDs from seed, split, family, and variant index. Skip rejected IDs and advance the variant index until the split quota is full.

- [ ] **Step 5: Implement bounded scene rendering**

For synthetic-only mode:

1. Load the source scene and select its largest valid plate anchor.
2. Resize the scene and anchor together so the longest edge is at most 960.
3. Render exactly one fictitious plate for a positive or inpaint the anchor for a negative.
4. Apply the assigned condition profile.
5. Save JPEG at configured quality and write the matching label atomically.
6. Export one OCR crop for every positive and store its relative path/checksum.
7. Reuse only when image, label, and required crop checksums all match.

Preserve the existing hybrid path for small legacy tests.

- [ ] **Step 6: Run GREEN and commit**

Run:

    .\.venv\Scripts\python -m pytest tests/test_split_builder.py tests/test_storage_ocr_export.py -v

Commit:

    git add plate_dataset/builder.py tests/test_split_builder.py
    git commit -m "feat: render resumable synthetic scenes with OCR crops"

### Task 5: Checkpointed parallel generation

**Files:**
- Modify: `plate_dataset/builder.py`
- Create: `tests/test_parallel_builder.py`

**Interfaces:**
- Consumes: deterministic `GenerationSpec` items.
- Produces: ordered checkpointed rendering with `workers=0` selecting `max(1, min(cpu_count - 1, 8))`.

- [ ] **Step 1: Write failing serial/parallel equivalence test**

    def test_parallel_and_serial_builds_have_identical_ids_and_labels(tmp_path):
        serial = build_dataset(replace(config, workers=1), records, tmp_path / "serial")
        parallel = build_dataset(replace(config, workers=2), records, tmp_path / "parallel")
        assert manifest_projection(serial) == manifest_projection(parallel)

- [ ] **Step 2: Write failing checkpoint test**

Inject a renderer that raises after seven results and assert the first five-row checkpoint exists. Rerun with the real renderer and assert those five outputs are reused while the requested total is reached.

- [ ] **Step 3: Run RED**

Run:

    .\.venv\Scripts\python -m pytest tests/test_parallel_builder.py -v

Expected: the current builder is serial and writes its manifest only at the end.

- [ ] **Step 4: Implement bounded process execution and checkpoints**

Use `ProcessPoolExecutor` with top-level picklable worker functions. Submit at most `workers * 2` pending items, consume completed results, sort manifest rows by output ID, and atomically checkpoint every 100 outputs in production and every five outputs in tests. Keep progress callbacks in the coordinator process. A worker failure records the output ID and exception type before stopping; the next run resumes checksum-valid checkpointed rows.

- [ ] **Step 5: Run GREEN and commit**

Run:

    .\.venv\Scripts\python -m pytest tests/test_parallel_builder.py tests/test_split_builder.py -v

Commit:

    git add plate_dataset/builder.py tests/test_parallel_builder.py
    git commit -m "perf: checkpoint parallel synthetic rendering"

### Task 6: Synthetic-only validation and QA reporting

**Files:**
- Modify: `plate_dataset/validate.py`
- Modify: `plate_dataset/reporting.py`
- Modify: `tests/test_validation_reporting.py`

**Interfaces:**
- Consumes: the extended manifest, OCR files, and exact generation quotas.
- Produces: zero-error validation only when every approved acceptance gate passes.

- [ ] **Step 1: Write failing synthetic validation tests**

Add fixtures asserting:

    def test_synthetic_mode_does_not_require_real_holdout(synthetic_fixture):
        report = validate_dataset(synthetic_fixture.root, synthetic_fixture.config)
        assert not any(issue.code == "real_holdout_share" for issue in report.issues)

    def test_validator_requires_linked_crop_for_every_positive(synthetic_fixture):
        missing_crop = synthetic_fixture.positive_crop
        missing_crop.unlink()
        report = validate_dataset(synthetic_fixture.root, synthetic_fixture.config)
        assert any(issue.code == "missing_ocr_crop" for issue in report.issues)

    def test_validator_enforces_exact_negative_quota(synthetic_fixture):
        make_one_positive_label_empty(synthetic_fixture)
        report = validate_dataset(synthetic_fixture.root, synthetic_fixture.config)
        assert any(issue.code == "negative_count_mismatch" for issue in report.issues)

- [ ] **Step 2: Run RED**

Run:

    .\.venv\Scripts\python -m pytest tests/test_validation_reporting.py -v

Expected: real holdout is still enforced and OCR quota/link checks are absent.

- [ ] **Step 3: Implement approved validation gates**

In synthetic-only mode, require every manifest origin to equal `synthetic`, skip the 80% real holdout rule, verify exact per-split totals and negatives from `generation_quotas`, validate every positive OCR path/checksum, require at least 46,250 crops for the default build, and enforce state/layout/condition proportions. Keep pair, box, checksum, family, and exact-duplicate checks.

Generate contact sheets grouped by split, plate style, layout, and condition. Captions contain output ID, category, and condition but never registration text.

- [ ] **Step 4: Run GREEN and commit**

Run:

    .\.venv\Scripts\python -m pytest tests/test_validation_reporting.py -v

Commit:

    git add plate_dataset/validate.py plate_dataset/reporting.py tests/test_validation_reporting.py
    git commit -m "feat: validate exact synthetic dataset quotas"

### Task 7: CLI, source normalization, and documentation

**Files:**
- Modify: `scripts/convert_annotations.py`
- Modify: `scripts/generate_synthetic.py`
- Modify: `scripts/validate_dataset.py`
- Modify: `README.md`
- Modify: `tests/test_cli.py`
- Remove: `tests/test_notebook.py` (untracked interrupted training-scope test)

**Interfaces:**
- Consumes: storage preflight, builder progress callback, rejection IDs, and synthetic validator.
- Produces: complete source-to-dataset commands without a training dependency.

- [ ] **Step 1: Write failing CLI tests**

Add assertions that generation help includes `--workers`, `--reject-file`, `--resume`, and `--no-resume`; dry-run reports 50,000 target images, exact split counts, projected storage, and creates no detector files.

- [ ] **Step 2: Run RED**

Run:

    .\.venv\Scripts\python -m pytest tests/test_cli.py -v

Expected: `--workers` and quota/storage dry-run output are missing.

- [ ] **Step 3: Implement source and generation CLI changes**

Make annotation conversion retain only allowed-source archives and write normalized records atomically. Add `--workers N` to generation, run storage preflight before creating output directories, pass parsed rejection IDs to the builder, print a progress line at each 500 completed outputs, merge existing and synthetic OCR labels after generation, and return nonzero when validation fails.

- [ ] **Step 4: Update README and remove obsolete training-scope test**

Document exact 50,000 counts, estimated disk use, commands, resume behavior, source licenses, all-synthetic holdout limitation, Colab upload instructions, and Camera Module 3 evaluation caveat. Remove only the known untracked `tests/test_notebook.py` created during the interrupted training task; do not remove user files.

- [ ] **Step 5: Run GREEN and commit**

Run:

    .\.venv\Scripts\python -m pytest -v
    git diff --check

Commit:

    git add scripts README.md tests/test_cli.py
    git commit -m "docs: expose 50000-image generation workflow"

### Task 8: Acquire sources, generate 50,000 outputs, and perform final QA

**Files generated and ignored by Git:**
- `raw/`
- `detection/images/{train,val,test}/`
- `detection/labels/{train,val,test}/`
- `ocr/images/{train,val,test}/`
- `ocr/labels.csv`
- `metadata/source_manifest.csv`
- `metadata/normalized_records.jsonl`
- `metadata/generation_manifest.csv`
- `metadata/dataset_statistics.json`
- `reports/validation_report.json`
- `reports/contact_sheets/*.jpg`

- [ ] **Step 1: Run the full pre-mutation verification**

Run:

    .\.venv\Scripts\python -m pytest -v
    .\.venv\Scripts\python scripts/download_sources.py --config config/default.yaml --dry-run
    .\.venv\Scripts\python scripts/generate_synthetic.py --config config/default.yaml --dry-run

Expected: tests pass, only the two allowed CC0 sources are eligible, and projected output preserves at least 5 GB free.

- [ ] **Step 2: Download and normalize allowed sources**

Run:

    $env:KAGGLE_CONFIG_DIR = 'D:\obj_det_dataset'
    .\.venv\Scripts\python scripts/download_sources.py --config config/default.yaml
    .\.venv\Scripts\python scripts/convert_annotations.py --config config/default.yaml

Verify that `metadata/source_manifest.csv` contains no credential path/value, all included sources have allowed decisions, and normalized records contain valid class-`0` boxes in all three source-family pools.

- [ ] **Step 3: Run a 100-output bounded integration build**

Create `config/smoke-100.yaml` with the approved output profile and target 100, run generation and validation into `reports/integration-100/`, inspect all generated contact sheets, then commit only the smoke configuration if it is retained.

- [ ] **Step 4: Generate exactly 50,000 outputs**

Run:

    .\.venv\Scripts\python scripts/generate_synthetic.py --config config/default.yaml --resume --workers 0

Keep the process resumable through manifest checkpoints. Do not start a second generator while this command is active.

- [ ] **Step 5: Validate and inspect visual samples**

Run:

    .\.venv\Scripts\python scripts/validate_dataset.py --config config/default.yaml --samples-per-sheet 25

Confirm `error_count` is zero. Inspect every generated contact sheet. Record implausible output IDs one per line in `reports/rejected_output_ids.txt`, rerun generation with `--reject-file`, and rerun validation until the report returns zero errors and no visually rejected IDs remain.

- [ ] **Step 6: Run final evidence collection**

Run:

    .\.venv\Scripts\python -m pytest -v
    .\.venv\Scripts\python scripts/validate_dataset.py --config config/default.yaml
    git diff --check
    git status --short

Record detector counts by split, negative count, MH share, OCR counts, source decisions, validation errors, generation seed, disk footprint, and free disk space in `README.md` and compact JSON reports.

- [ ] **Step 7: Commit compact reproducibility metadata**

Run:

    git add README.md metadata/dataset_statistics.json reports/validation_report.json
    git commit -m "data: validate 50000-image synthetic plate dataset"

Generated bulk data remains ignored and uncommitted.
