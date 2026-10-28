import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .common import ROOT, digest, probabilities, read_json, resolve_path, write_json
from .data import validate_manifest
from .evaluate import make_report, metrics
from .risk import certify, conformal_policy, fit_temperature, tune_policy


def load_predictions(path, manifest, split):
    with np.load(path, allow_pickle=False) as z:
        obj = {k: z[k] for k in z.files}
    expected = manifest[manifest.split == split].sort_values('image_id')
    if not np.array_equal(obj['image_id'], expected.image_id.to_numpy(dtype=str)):
        raise ValueError(f'Prediction IDs/order do not match frozen {split} split')
    if not np.array_equal(obj['y'], expected.y.to_numpy(dtype=np.int64)):
        raise ValueError('Prediction labels mismatch')
    if len(obj['p']) != len(expected):
        raise ValueError('Prediction length mismatch')
    probabilities(obj['p'])
    return obj


def freeze_experiment(cfg):
    out = resolve_path(cfg['output_dir'])
    code = {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted((ROOT/'poultryrisk').glob('*.py'))}
    spec = {'config': cfg, 'manifest_sha256': digest(out/'data/manifest.csv'), 'source_sha256': code}
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    spec['fingerprint'] = fingerprint
    target = out/'experiment_freeze.json'
    if target.exists():
        if read_json(target)['fingerprint'] != fingerprint:
            raise ValueError('Config, manifest, or source changed. Use a NEW experiment/output_dir; preserve prior results.')
    else:
        write_json(target, spec)
        write_json(out/'environment.json', {'python': sys.version, 'platform': platform.platform(),
           'pip_freeze': subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True).stdout.splitlines()})


def verify_policy_freeze(out):
    verify_selection_freeze(out)
    record = read_json(out/'policy_freeze.json')
    for name, sha in record['files'].items():
        if digest(out/name) != sha:
            raise ValueError(f'Policy/calibration inputs changed: {name}')


def verify_selection_freeze(out):
    record = read_json(out/'selection_freeze.json')
    for name, sha in record['files'].items():
        if digest(out/name) != sha:
            raise ValueError(f'Validation selection inputs changed: {name}')


def select_run(frame, cfg, out):
    """Freeze validation-only candidates before creating calibration predictions."""
    if (out/'selection_freeze.json').exists():
        verify_selection_freeze(out)
        return
    if list(out.glob('predictions_*_calibration.npz')) or list(out.glob('predictions_*_test.npz')):
        raise ValueError('Calibration/test predictions already exist without selection freeze. Use a NEW experiment.')
    selected, tracked = {}, []
    for fmt in cfg['formats']:
        name = f'predictions_{fmt}_val.npz'
        val = load_predictions(out/name, frame, 'val')
        temperature = fit_temperature(val['p'], val['y'])
        selected[fmt] = {
            'temperature': temperature,
            'global': tune_policy(val['p'], val['y'], cfg, temperature=temperature),
            'proposed': tune_policy(val['p'], val['y'], cfg, classwise=True, temperature=temperature),
        }
        tracked.append(name)
        for artifact in [f'model_{fmt}.tflite', f'export_{fmt}.json']:
            if (out/artifact).exists():
                tracked.append(artifact)
    write_json(out/'selected_policies.json', selected)
    tracked.append('selected_policies.json')
    write_json(out/'selection_freeze.json', {
        'files': {name: digest(out/name) for name in tracked},
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'calibration_used': False, 'test_used': False,
    })


def calibrate_run(frame, cfg, out):
    verify_selection_freeze(out)
    if (out/'policy_freeze.json').exists():
        verify_policy_freeze(out)
        return
    family = 2*len(cfg['formats'])*len(cfg['backbones'])*len(cfg['seeds'])
    selected = read_json(out/'selected_policies.json')
    policies, certificates, tracked = {}, {}, ['selection_freeze.json', 'selected_policies.json']
    for fmt in cfg['formats']:
        vf, cf = f'predictions_{fmt}_val.npz', f'predictions_{fmt}_calibration.npz'
        cal = load_predictions(out/cf, frame, 'calibration')
        temperature = selected[fmt]['temperature']
        global_rule = selected[fmt]['global']
        proposed = selected[fmt]['proposed']
        cg = certify(cal['p'], cal['y'], global_rule, cfg, family)
        cp = certify(cal['p'], cal['y'], proposed, cfg, family)
        policies[fmt] = {
            'maxsoftmax_090': {'kind': 'threshold', 'thresholds': [0.9]*4, 'margin': 0.0, 'temperature': 1.0},
            'temperature_validation_rule': global_rule,
            'class_conditional_conformal': conformal_policy(cal['p'], cal['y'], cfg['alpha'], temperature),
            'global_certified': cg['deployed_policy'],
            'proposed_certified': cp['deployed_policy'],
            'proposed_uncertified_ablation': proposed,
        }
        certificates[fmt] = {'global': cg, 'proposed': cp}
        tracked.extend([vf, cf])
    if 'int8' in policies:
        # Diagnostic baseline only: this transferred policy receives NO INT8 certificate.
        policies['int8']['fp32_rule_transferred_unverified'] = dict(policies['fp32']['proposed_uncertified_ablation'])
    write_json(out/'policies.json', policies)
    write_json(out/'certificates.json', certificates)
    tracked.extend(['policies.json', 'certificates.json'])
    for fmt in cfg['formats']:
        for name in [f'model_{fmt}.tflite', f'export_{fmt}.json']:
            if (out/name).exists():
                tracked.append(name)
    write_json(out/'policy_freeze.json', {'files': {p: digest(out/p) for p in tracked},
        'family_size': family, 'delta': cfg['delta'],
        'test_used': False, 'claim': 'conditional on documented statistical assumptions; no clinical validation'})


def evaluate_run(frame, cfg, out):
    from .models import infer
    verify_policy_freeze(out)
    policies = read_json(out/'policies.json')
    for fmt in cfg['formats']:
        infer(frame, cfg, out, fmt, 'test')
        test = load_predictions(out/f'predictions_{fmt}_test.npz', frame, 'test')
        make_report(test['y'], test['p'], policies[fmt], out/f'evaluation_{fmt}', cfg)
        by_source = {}
        for source in sorted(set(test['source'])):
            mask = test['source'] == source
            by_source[str(source)] = {name: metrics(test['y'][mask], test['p'][mask], policy)
                                     for name, policy in policies[fmt].items()}
        write_json(out/f'evaluation_{fmt}/source_subgroups.json', {
            'scope': 'subgroups of the same mixed-source holdout; NOT cross-source generalization or source-specific certification',
            'groups': by_source})


def run(cfg, stage):
    from . import models
    if cfg.get('evaluation_role') == 'development' and stage not in {'train', 'export'}:
        raise ValueError('Development config may only train/export. Keep calibration/test unopened; use full.yaml after protocol freeze.')
    frame = validate_manifest(cfg, verify_bytes=True)
    freeze_experiment(cfg)
    for backbone in cfg['backbones']:
        for seed in cfg['seeds']:
            out = resolve_path(cfg['output_dir']) / backbone / f'seed_{seed}'
            out.mkdir(parents=True, exist_ok=True)
            print(f'Running {backbone} seed={seed} stage={stage}', flush=True)
            if stage in {'train', 'all'}:
                models.train(frame, cfg, backbone, seed, out)
            if stage in {'export', 'all'}:
                models.export(frame, cfg, out)
            if stage in {'select-policy', 'predict-calibration', 'all'}:
                for fmt in cfg['formats']:
                    models.infer(frame, cfg, out, fmt, 'val')
                select_run(frame, cfg, out)
            if stage in {'predict-calibration', 'all'}:
                for fmt in cfg['formats']:
                    models.infer(frame, cfg, out, fmt, 'calibration')
            if stage in {'calibrate', 'all'}:
                calibrate_run(frame, cfg, out)
            if stage in {'evaluate', 'all'}:
                evaluate_run(frame, cfg, out)
    if stage in {'evaluate', 'all'}:
        collect(cfg)


def collect(cfg):
    root = resolve_path(cfg['output_dir'])
    frames = []
    for backbone in cfg['backbones']:
        for seed in cfg['seeds']:
            for fmt in cfg['formats']:
                runout = root/backbone/f'seed_{seed}'
                path = runout/f'evaluation_{fmt}/summary.csv'
                if path.exists():
                    f = pd.read_csv(path)
                    f.insert(0, 'backbone', backbone)
                    f.insert(1, 'seed', seed)
                    f.insert(2, 'format', fmt)
                    cert = read_json(runout/'certificates.json')[fmt]
                    f['certificate_status'] = f['method'].map({
                        'global_certified': cert['global']['status'],
                        'proposed_certified': cert['proposed']['status']}).fillna('NOT_THIS_CERTIFICATE')
                    frames.append(f)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(root/'all_results.csv', index=False)


def doctor():
    import importlib.metadata
    result = {'python': sys.version, 'platform': platform.platform(), 'packages': {}}
    for package in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'Pillow', 'PyYAML', 'tensorflow']:
        try:
            result['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result['packages'][package] = 'NOT INSTALLED'
    if result['packages']['tensorflow'] != 'NOT INSTALLED':
        from .models import tensorflow
        tf = tensorflow()
        result['gpu_devices'] = [x.name for x in tf.config.list_physical_devices('GPU')]
    print(json.dumps(result, indent=2))
