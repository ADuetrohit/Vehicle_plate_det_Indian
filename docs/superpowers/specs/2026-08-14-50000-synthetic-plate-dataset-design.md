# 50,000-Image Indian Number-Plate Synthetic Dataset Design

## Objective

Generate exactly 50,000 synthetic full-scene Indian number-plate detection images under `D:\obj_det_dataset\synthetic_number_plate_dataset`. Every scene will have a matching YOLO label file. Every positive scene will also yield an OCR crop with the exact fictitious registration text used during rendering. Full detector training remains the user's responsibility in Google Colab.

The final outputs are synthetic because every visible registration plate is newly rendered or deliberately removed. License-approved real images are used only as geometry and scene anchors; unmodified source images are not copied into the final detector dataset.

## Fixed Counts

- Detector images: exactly 50,000.
- Detector labels: exactly 50,000.
- Train split: 40,000 image-label pairs.
- Validation split: 5,000 image-label pairs.
- Test split: 5,000 image-label pairs.
- Hard negatives: 3,750 scenes, distributed proportionally as 3,000 train, 375 validation, and 375 test.
- Positive detector scenes: 46,250.
- Synthetic OCR crops: at least 46,250, with one crop for each rendered visible plate.
- Existing reviewed OCR crops: preserve the separate 2,122-crop import and do not count it toward the 46,250 synthetic minimum.

## Source Policy

Use only the two source-registry datasets marked `allowed` and recorded as CC0:

1. `kedarsai/indian-license-plates-with-labels`
2. `deepakat002/indian-vehicle-number-plate-yolo-annotation`

Datasets marked `verify` or `blocked` remain excluded unless a separate, explicit license decision is made. Kaggle credentials are read only from `KAGGLE_CONFIG_DIR`; credential values must never appear in source code, console output, manifests, reports, or commits.

The source stage must checksum downloaded archives, extract them safely, normalize all valid plate boxes to class `0`, group exact and perceptual duplicates, and retain the source dataset and source-family identifier.

## Generation Architecture

### Family-safe split

Assign unique source families to train, validation, or test before creating variants. No base image, near duplicate, or derived synthetic variant may cross split boundaries. Allocate approximately 80%, 10%, and 10% of available source families to the three splits, then generate exactly the fixed output counts within each split.

Because the requested final package is fully synthetic, the earlier requirement that validation and test be 80% unmodified real images is intentionally replaced. Validation and test still use disjoint real-scene source families, but all visible plates in their final images are synthetic. The validation report and README must state this limitation.

### Registration identity

Generate fictitious format-valid Indian registrations. Target 65% Maharashtra registrations among positive scenes, accepting any final value from 60% through 70%. Do not reproduce registration text from source metadata or the existing OCR collection. Normalize all OCR labels to uppercase alphanumeric compact text.

### Plate appearance

Include white private, yellow commercial, green electric private, green electric commercial, and temporary plates. Target 20% double-row layouts, accepting 15% through 30%. Vary font, spacing, border, screws, surface wear, dirt, and mild damage while retaining a clean high-resolution master for OCR.

### Scene compositing

Use one primary valid source plate region as the anchor for each positive output. Perspective-warp the rendered plate into that region, alpha composite it into the scene, and recompute a tight class-`0` axis-aligned bounding box from transformed corners. For a hard negative, inpaint the anchor and emit an empty matching YOLO label file.

Create balanced variants across day, night, rain, fog, glare, shadow, motion blur, compression, brightness, noise, distance, and occlusion. Target 20% through 30% night or low-light scenes and at least 15% adverse-condition scenes. Limit degradations so detector boundaries remain visible; a scene may remain detector-eligible while its crop is marked OCR-ineligible.

### Image size and storage

Normalize full scenes without distorting their aspect ratio. Limit the longest edge to 960 pixels, retain enough resolution for 512-pixel YOLO training, and encode JPEG at quality 88. Encode OCR crops on a 256×128 canvas with aspect-preserving padding. Use a preflight storage estimate and refuse to start when projected final output plus raw sources and temporary headroom exceeds available disk.

The current D: drive has 41.83 GB free before source acquisition. The build should target a final footprint below 25 GB and preserve at least 5 GB of free space.

## Output Structure

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

Bulk outputs remain ignored by Git. Code, configuration, tests, documentation, license decisions, and compact reports remain versioned.

## Metadata and Resume

Derive every output ID and random stream from seed `20260814`, split, source family, and variant index. Write each image, label, and crop through a temporary sibling followed by atomic rename.

The generation manifest records:

- output ID and split;
- origin and source family;
- relative image, label, and OCR-crop paths;
- SHA-256 values;
- negative flag;
- fictitious plate text and state;
- plate category and layout;
- vehicle/viewpoint metadata when known;
- applied condition profile;
- detector and OCR eligibility.

On rerun, reuse an output only when all required files exist and their checksums match the manifest. A rejected output ID from visual QA is replaced with the next deterministic variant without deleting unrelated files.

## Validation

The final validator must confirm:

- exactly 50,000 readable detector images and 50,000 matching labels;
- exact split counts of 40,000, 5,000, and 5,000;
- exactly 3,750 intentional empty-label hard negatives;
- no missing image-label pairs or orphan files;
- every non-empty row has five finite values, class `0`, positive dimensions, and coordinates contained in `[0,1]`;
- every positive plate remains at least 8×4 pixels at 512-pixel letterboxed training size;
- image, label, and crop checksums match the manifest;
- no source family or exact duplicate crosses splits;
- Maharashtra share is between 60% and 70% of positive scenes;
- double-row share is between 15% and 30%;
- low-light and adverse-condition shares meet their target ranges;
- every OCR-eligible positive has a readable crop and exact text label;
- at least 46,250 synthetic OCR crops exist;
- the 2,122 existing OCR crops remain preserved and separately identified;
- generated contact sheets cover every split, plate style, layout, and major condition.

Validation succeeds only with zero errors. Warnings may identify unknown source vehicle/viewpoint metadata, but must not be silently converted into invented labels.

## Error Handling

- Abort a source before inclusion when its license is not allowed.
- Reject unsafe archives, corrupted images, invalid annotations, and undersized anchors with an explicit reason.
- Continue past individual rejected source records when sufficient valid families remain.
- Stop before generation if no valid source family exists in any split.
- Stop when storage preflight predicts less than 5 GB remaining.
- Preserve all existing workspace archives, models, credentials, CSV files, and the 2,122 imported OCR crops.
- Never remove or overwrite unrelated user files.

## Testing Strategy

Use test-first changes for the scale-up:

1. Configuration tests require a 50,000 target and fixed split/negative counts.
2. Split tests confirm family isolation for fully synthetic holdouts.
3. Builder tests confirm exact quotas, bounded dimensions, OCR crops, stable IDs, and checksum resume.
4. Storage tests confirm a build is rejected before mutation when capacity is insufficient.
5. Validation tests confirm exact counts, OCR linkage, distribution gates, and zero cross-split families.
6. A bounded integration fixture exercises download-independent generation before the full run.

After implementation, run the complete unit suite, generate all outputs, run full validation, inspect representative contact sheets, and rerun validation after any visual rejection replacements.

## Acceptance Criteria

The request is complete only when exactly 50,000 detector image-label pairs and at least 46,250 synthetic OCR crops exist inside the project, full validation reports zero errors, split and distribution counts satisfy this document, representative contact sheets pass visual inspection, final disk usage is recorded, and reproducibility commands are documented.

Training a detector, exporting a model, and benchmarking real Camera Module 3 footage are not part of this generation request.
