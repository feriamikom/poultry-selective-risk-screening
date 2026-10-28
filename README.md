# Poultry selective risk screening

A Comparative Evaluation of Selective FP32 and INT8 Poultry Fecal-Image Classifiers under Classwise Risk Screening

This empirical study compares selective FP32 and INT8 MobileNetV2 and
EfficientNetB0 classifiers on four poultry fecal-image labels. The manuscript is
**under review**. The original primary family contains 24 selected rules and 96
class bounds: 7 passed and 17 remain failed with all-defer deployment. All
analyses added during peer review are post hoc and exploratory; they do not
replace that record.
This package does not establish clinical utility, methodological novelty, or
journal acceptance.

## Contents

- `poultryrisk/`: byte-identical frozen training, export and risk-screening code.
- `revision/`, `scripts/revision_run_all.py`: deterministic saved-prediction analyses.
- `artifacts/full_colab_v1/`: data-cleaning audit, splits, predictions, policies,
  certificates, freezes, training/export metadata and original host latency summaries.
- `artifacts/revision_ijies_20267101/`: tables and figures of the revised
  manuscript and supplement, hash manifest, hash-bound graph-inspection cache and
  the documented-host latency benchmark (`latency_host_documented/`).
- `scripts/revision_latency_benchmark.py`: the post hoc host latency benchmark.
- `docs/`: research protocol, statistical contract, the post hoc IUT explanation
  and verified method sources.
- `tests/`, `configs/`, `notebooks/`: verification, original settings and a clean
  unexecuted-output copy of the Colab notebook.

No dataset image, raw archive, manuscript, review or credential is included in this
repository. Model weights are not stored in the repository itself; they are attached
to the GitHub release (see Pretrained weights). The code license below does not
license the dataset.

## Reproduce the manuscript tables without dataset images or weights

Use an isolated Python 3.11 environment. On Linux or existing WSL2:

```bash
python3.11 -m venv .venv-revision
.venv-revision/bin/python -m pip install -r requirements-core.txt
.venv-revision/bin/python -m unittest discover -s tests -v
.venv-revision/bin/python scripts/statistical_smoke.py
.venv-revision/bin/python scripts/revision_run_all.py \
  --input-root artifacts/full_colab_v1 \
  --output-root results/reproduced_revision \
  --allow-missing-models --phase all
```

The core requirements suffice for these saved-artifact analyses. Native operator
inspection used TensorFlow 2.20.0; artifact-only reproduction reads the saved
inspection cache and checks each recorded model SHA256 against export metadata.
It does not re-inspect absent model binaries. The provenance snapshot preserves
historical extraction timestamps and original stage-log hashes; copied filesystem
timestamps are not treated as evidence of prior chronology. The visual-inspection
montage of cross-split image pairs requires the original images and is therefore
not included. Table CSV hashes can be compared to
`artifacts/revision_ijies_20267101/revision_manifest.json`. Execution logs and
the image montage are not part of these deterministic outputs.

## Dataset access and full pipeline reproduction

The images originate from the collections
[Zenodo 4628934](https://zenodo.org/records/4628934) and
[Zenodo 5801834](https://zenodo.org/records/5801834). Consult those records and
their authors for reuse permission: no dataset license was displayed when the
records were checked in October 2026. Images are not redistributed here. Filename prefixes are unverified
source proxies, not verified PCR provenance or animal/flock identifiers.

The archive used in the study (`archive.zip`) had SHA256
`6641a21b18310e23c1b848c99c0552b66eaf58bb3821f52563aecec448e0704e`.
Repacking downloads changes archive bytes and cannot establish this byte-level
identity. Obtain the original archive from an authorized source and verify it
locally before claiming exact data reproduction. The retained manifest SHA256 is
`712a8f8782e77ff7d2f77e29d826fa0bd3187fccff7ecb973f3cf6d0450db494`.

```bash
sha256sum archive.zip
.venv-revision/bin/python -m pip install -r requirements.txt
.venv-revision/bin/python -m poultryrisk doctor
.venv-revision/bin/python scripts/create_synthetic_dataset.py
.venv-revision/bin/python -m poultryrisk prepare --config configs/smoke.yaml
.venv-revision/bin/python -m poultryrisk run --config configs/smoke.yaml --stage all
.venv-revision/bin/python -m poultryrisk prepare --config configs/full.yaml --zip archive.zip
.venv-revision/bin/python -m poultryrisk verify-data --config configs/full.yaml
.venv-revision/bin/python -m poultryrisk feasibility --config configs/full.yaml
.venv-revision/bin/python -m poultryrisk run --config configs/full.yaml --stage all
```

The checked-in `full.yaml` writes a new `results/full_v1/` experiment; packaged
historical artifacts remain untouched. Keep completed freezes and use a new
experiment directory for changed code/configuration. Train uses train, selection
uses validation, screening uses calibration and final reporting uses test.
Representative INT8 images come only from train. Review the cleaning decisions
before training on real data. Native Windows TensorFlow uses CPU; GPU execution requires a
compatible existing Linux/WSL2 environment and its appropriate dependencies.
Historical resolved dependencies are in `results/colab_dependencies.lock.txt`;
direct dependency pins alone do not guarantee bit-identical retraining.

## Interpretation and limits

Class order is healthy, coccidiosis, salmonella, newcastle. Automatic error per
true class uses all images of that class; selective error uses accepted images
and is null for all-defer. Report coverage with risk. The confidence statements
depend on within-class independence/exchangeability, which this image audit cannot
verify. Seed replication reuses the same images and does not enlarge calibration
or test sample size. No guarantee automatically transfers under distribution
shift. Class-conditional conformal coverage, expected CRC risk and high-probability
risk screening are different statements.

Latency is reported for host CPUs only. The original Colab summaries
(`artifacts/full_colab_v1/*/seed_*/latency_*.json`) give median and p95 invoke
times; their CPU model and raw per-image times were not saved. A post hoc
benchmark of the same 12 TFLite files on a documented desktop CPU (Intel Core
i3-10105F under WSL2, TensorFlow 2.20.0) is in
`artifacts/revision_ijies_20267101/latency_host_documented/`: raw per-image
invoke and end-to-end times for 1 and 2 interpreter threads, summary statistics,
process RSS, host information, and a check that the outputs equal the saved
predictions. Reproducing it requires the original archive and the model release:

```bash
.venv-revision/bin/python scripts/revision_latency_benchmark.py
```

No smartphone/embedded latency, device memory, hardware-level integer execution,
field effectiveness, or clinical safety claim follows from these data.

## Pretrained weights

The release asset `models_full_colab_v1.zip` contains the six Keras checkpoints
(`best.keras`), the twelve TFLite exports (`model_fp32.tflite`, `model_int8.tflite`),
`SHA256SUMS.txt` and `WEIGHTS_NOTICE.md`. Place its contents under
`artifacts/full_colab_v1/` (same relative paths) and verify them with
`sha256sum -c SHA256SUMS.txt`; each hash also matches the frozen
`export_*.json`. The weights are released for research reproducibility. They were
trained on the Zenodo collections above; users must cite and respect the terms of
the dataset creators. The MIT license covers the code only.

## Citation and license

See `CITATION.cff` for citation metadata; a DOI will be added after publication.
Code is MIT licensed, copyright 2026 Nafiatun Sholihah, Majid Rahardi, Sundari, Ferian Fauzi Abdulloh. Correspondence: nafiatun@amikom.ac.id.
