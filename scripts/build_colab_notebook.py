"""Build the local Colab handoff notebook. Does not upload or execute cells."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def cell(kind, text):
    item = {'cell_type': kind, 'metadata': {}, 'source': text.strip().splitlines(keepends=True)}
    if kind == 'code':
        item.update(execution_count=None, outputs=[])
    cells.append(item)


cell('markdown', '''
# Nafi BIMA — Colab preparation and development pilot

Run cells individually when working hours begin (08:30 WIB or later).
This notebook was prepared locally; it has not been executed on Colab.
Do not use **Run all**: image-cleaning review is required before the pilot.

1. Upload this notebook through Colab's **File → Upload notebook**.
2. Choose a GPU runtime in **Runtime → Change runtime type**. Check the available
   GPU, your compute-unit balance, and the usage rate shown by Colab.
3. In your own Drive, manually create `Nafi_BIMA_RiskControl` and place the
   source-only `nafi_colab_source.zip` and the confirmed `archive.zip` there.
   No files were uploaded during local preparation.
4. Mount Drive below. Source and extracted images use the VM disk; results and
   checkpoints go to the dedicated Drive folder for recovery after disconnects.

Compute-unit use and GPU availability vary. Measure the pilot before estimating
the main study's budget. See the [official Colab FAQ](https://research.google.com/colaboratory/faq.html).
Disconnect/delete the runtime when finished; closing a tab alone is not the
workflow for releasing a runtime.
''')
cell('code', '''
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, time, zipfile
from datetime import datetime, timezone

# Colab exports an interactive Matplotlib backend for its kernel.  The isolated
# project environment intentionally does not depend on matplotlib-inline, so
# force a headless backend for every subprocess launched from this notebook.
os.environ['MPLBACKEND'] = 'Agg'

DRIVE_PROJECT = Path('/content/drive/MyDrive/Nafi_BIMA_RiskControl')
SOURCE_ZIP = DRIVE_PROJECT / 'nafi_colab_source.zip'
DATASET_ZIP = DRIVE_PROJECT / 'archive.zip'
PROJECT = Path('/content/Nafi_BIMA_RiskControl')
RESULTS = DRIVE_PROJECT / 'results'
assert SOURCE_ZIP.is_file(), f'Manually place the source package at {SOURCE_ZIP}'
assert DATASET_ZIP.is_file(), f'Manually place the confirmed dataset at {DATASET_ZIP}'
PROJECT.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

# Verify package bytes and reject unsafe entries or replacement of changed source.
with zipfile.ZipFile(SOURCE_ZIP) as archive:
    manifest = json.loads(archive.read('package_manifest.json'))
    names = archive.namelist()
    assert len(names) == len(set(names)), 'Duplicate ZIP entry'
    assert set(names) == set(manifest) | {'package_manifest.json'}
    for name, sha in manifest.items():
        target = (PROJECT / name).resolve()
        assert target.is_relative_to(PROJECT.resolve()), 'Unsafe source path'
        info = archive.getinfo(name)
        assert (info.external_attr >> 16) & 0o170000 != 0o120000, 'Symlink rejected'
        content = archive.read(name)
        assert hashlib.sha256(content).hexdigest() == sha, name
        if target.exists():
            assert target.read_bytes() == content, f'Changed source: {name}; preserve prior experiment first'
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
os.chdir(PROJECT)
SESSION = RESULTS / ('session_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
SESSION.mkdir()
(SESSION / 'source_package_manifest.json').write_text(json.dumps(manifest, indent=2))
print('Project:', PROJECT, '\\nPersistent results:', RESULTS)
''')
cell('markdown', '''
## Isolated Python 3.11 and pinned project requirements

Notebook cells use Colab's kernel only to launch commands. Scientific commands
use the explicit isolated interpreter below, not Colab's preinstalled packages.
This installs the project's Linux GPU requirements; it does not change drivers.
''')
cell('code', '''
def command(argv, label):
    log = SESSION / (label + '.log')
    record = {'argv': [str(x) for x in argv], 'started_utc': datetime.now(timezone.utc).isoformat()}
    (SESSION / (label + '.json')).write_text(json.dumps(record, indent=2))
    begin = time.monotonic()
    with log.open('w') as handle:
        process = subprocess.Popen([str(x) for x in argv], cwd=PROJECT,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end='')
            handle.write(line)
            handle.flush()
        code = process.wait()
    record.update(exit_code=code, elapsed_seconds=time.monotonic()-begin)
    (SESSION / (label + '.json')).write_text(json.dumps(record, indent=2))
    if code:
        raise RuntimeError(f'{label} failed ({code}); inspect {log}. Do not continue dependent stages.')

command([sys.executable, '-m', 'pip', 'install', 'uv==0.12.17'], 'bootstrap_uv')
command([sys.executable, '-m', 'uv', 'python', 'install', '3.11.16'], 'install_python')
PY = PROJECT / '.venv/bin/python'
if not PY.exists():
    command([sys.executable, '-m', 'uv', 'venv', '--python', '3.11.16', str(PROJECT / '.venv')], 'create_venv')
LOCK = RESULTS / 'colab_dependencies.lock.txt'
requirements = str(LOCK) if LOCK.exists() else 'requirements-wsl-gpu.txt'
command([sys.executable, '-m', 'uv', 'pip', 'install', '--python', str(PY),
         'pip==24.0', '-r', requirements], 'install_requirements')
command([PY, '-m', 'pip', 'check'], 'pip_check')
command([PY, '-m', 'pip', 'freeze'], 'dependencies')
resolved = subprocess.check_output([str(PY), '-m', 'pip', 'freeze'], text=True)
(SESSION / 'dependencies.txt').write_text(resolved)
if LOCK.exists():
    assert sorted(LOCK.read_text().splitlines()) == sorted(resolved.splitlines()), 'Environment differs from saved dependency lock'
else:
    LOCK.write_text(resolved)
command(['nvidia-smi'], 'gpu_inventory')
command([PY, '-m', 'poultryrisk', 'doctor'], 'doctor')
command([PY, '-c', 'import tensorflow as tf; g=tf.config.list_physical_devices("GPU"); print(g); assert g, "TensorFlow cannot see the GPU; resolve before training"'], 'gpu_check')
''')
cell('markdown', '''
## Preparation checks and persistent configurations

No real-data training in this cell. Configurations keep the declared scientific
parameters. Only experiment names and output storage change for Colab. Existing
configs must match; never overwrite a frozen experiment with changed source.
''')
cell('code', '''
command([PY, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], 'unit_tests')
command([PY, 'scripts/statistical_smoke.py'], 'statistical_smoke')
config_script = """
from pathlib import Path
import yaml
for stem in ['smoke', 'pilot', 'full']:
    cfg = yaml.safe_load(Path(f'configs/{stem}.yaml').read_text())
    cfg['experiment'] = stem + '_colab_v1'
    cfg['output_dir'] = str(Path('/content/drive/MyDrive/Nafi_BIMA_RiskControl/results') / cfg['experiment'])
    path = Path(f'configs/colab_{stem}.yaml')
    if path.exists():
        assert yaml.safe_load(path.read_text()) == cfg, 'Existing Colab config differs'
    else:
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
"""
command([PY, '-c', config_script], 'colab_configs')
SMOKE = 'configs/colab_smoke.yaml'
PILOT = 'configs/colab_pilot.yaml'
FULL = 'configs/colab_full.yaml'

def cli(action, config, *extra, label):
    command([PY, '-m', 'poultryrisk', action, '--config', config, *extra], label)

if not (PROJECT / 'data/synthetic').exists():
    command([PY, 'scripts/create_synthetic_dataset.py'], 'synthetic_generation')
if not (RESULTS / 'smoke_colab_v1/data/manifest.csv').exists():
    cli('prepare', SMOKE, label='synthetic_prepare')
cli('verify-data', SMOKE, label='synthetic_verify')
cli('feasibility', SMOKE, label='synthetic_feasibility')
''')
cell('markdown', '''
## Training gate 1: artificial-image TensorFlow/INT8 smoke

This cell trains TinyCNN for one epoch and executes the synthetic pipeline.
Run it only when ready to begin training. Resolve any failure here before
opening the real dataset. Synthetic results are not poultry research evidence.
''')
cell('code', '''
cli('run', SMOKE, '--stage', 'all', label='tensorflow_smoke')
assert (RESULTS / 'smoke_colab_v1/all_results.csv').is_file()
SMOKE_PASSED_THIS_SESSION = True
''')
cell('markdown', '''
## Audit the confirmed dataset

This verifies the original local ZIP hash before copying it to the VM. It does
not use a replacement dataset. On reconnect, extract the same archive if the
VM-local images were lost, then verify the saved manifest rather than recreating
the split. The archive is copied as one file to limit Drive I/O.
''')
cell('code', '''
assert globals().get('SMOKE_PASSED_THIS_SESSION') is True, 'Complete the synthetic gate first'
EXPECTED_DATASET_SHA256 = '6641a21b18310e23c1b848c99c0552b66eaf58bb3821f52563aecec448e0704e'
def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()
assert sha256(DATASET_ZIP) == EXPECTED_DATASET_SHA256, 'Dataset differs from confirmed archive'
local_zip = PROJECT / 'archive.zip'
if not local_zip.exists():
    shutil.copy2(DATASET_ZIP, local_zip)
assert sha256(local_zip) == EXPECTED_DATASET_SHA256
raw = PROJECT / 'data/raw'
has_images = raw.exists() and any(p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'} for p in raw.rglob('*'))
if not has_images:
    command([PY, '-c', 'from poultryrisk.data import extract_zip; extract_zip("archive.zip", "data/raw")'], 'extract_dataset')
if not (RESULTS / 'pilot_colab_v1/data/manifest.csv').exists():
    cli('prepare', PILOT, label='pilot_prepare')
cli('verify-data', PILOT, label='pilot_verify')
cli('feasibility', PILOT, label='pilot_feasibility')
print((RESULTS / 'pilot_colab_v1/data/audit.json').read_text())
''')
cell('markdown', '''
## Stop to review cleaning before the pilot

Inspect rejected files, class counts, near-duplicate clusters and cross-label
conflicts, including image examples. Record the inspection in the project docs.
Do not tune deduplication using test performance. Filename prefixes remain source
proxies; licensing and label provenance still need verification.

Once that review is complete, execute the next two cells individually.
They use train/validation only. The development config blocks calibration/test.
''')
cell('code', "cli('run', PILOT, '--stage', 'train', label='pilot_train')")
cell('code', "cli('run', PILOT, '--stage', 'export', label='pilot_export')")
cell('markdown', '''
## Before the main study

Review pilot duration, actual GPU and consumed compute units. Do not extrapolate
cost from an assumed fixed GPU rate. Complete the literature/comparator and
sample-size review before freezing the main protocol. No main calibration/test
cells are executed by this starter notebook.

The prepared `configs/colab_full.yaml` preserves two backbones, three seeds,
FP32/INT8, alpha=0.10 and familywise delta=0.05 (24 certified policies, 96 bounds).
For later execution use the CLI stages in `docs/RUNBOOK_0830.md`, replacing the
local config paths with `configs/colab_*.yaml` and keeping output on Drive.
Train/export/select all configurations before calibration when running in stages.

After any interruption, retain all output files. Rerun compatible completed
stages; interrupted training restarts that training stage, not the last epoch.
Changing source/config after freeze requires a new experiment output directory.
An incompatible dependency change also requires a documented new experiment;
do not treat arbitrary package upgrades as an equivalent environment.

Copy final manifests, source/config snapshots, environment logs, checkpoints,
TFLite files, policies, certificates and complete result tables back to the
local project after the run. Record whether real calibration/test was opened.
Release the runtime through Colab's Runtime menu when done.
''')

notebook = {'nbformat': 4, 'nbformat_minor': 4, 'metadata': {
    'colab': {'name': 'Nafi_BIMA_Colab.ipynb'},
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
}, 'cells': cells}
target = ROOT/'notebooks/Nafi_BIMA_Colab.ipynb'
target.parent.mkdir(exist_ok=True)
target.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
for index, item in enumerate(cells):
    if item['cell_type'] == 'code':
        compile(''.join(item['source']), f'cell_{index}', 'exec')
print(f'Created and syntax-checked {target}; no cells executed.')
