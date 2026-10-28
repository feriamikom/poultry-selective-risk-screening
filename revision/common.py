"""Portable, hash-tracked readers and deterministic revision output helpers."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pandas as pd
from poultryrisk.pipeline import load_predictions

CLASSES = ['healthy', 'coccidiosis', 'salmonella', 'newcastle']
EVIDENCE = 'post-hoc exploratory; primary certificates unchanged'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def clean_json(obj):
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, np.generic):
        return clean_json(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, Path):
        return obj.as_posix()
    return obj


def class_counts(y, pred, accept):
    rows = []
    for c, label in enumerate(CLASSES):
        mask = y == c
        n = int(mask.sum())
        accepted = int((mask & accept).sum())
        errors = int((mask & accept & (pred != y)).sum())
        rows.append({'class': label, 'n': n, 'accepted': accepted, 'errors': errors,
                     'coverage': accepted/n if n else None,
                     'automatic_error': errors/n if n else None,
                     'selective_error': errors/accepted if accepted else None,
                     'false_healthy': int((mask & accept & (pred == 0)).sum()) if c else None})
    return rows


def summarize(y, pred, accept):
    rows = class_counts(y, pred, accept)
    accepted = int(accept.sum())
    errors = int((accept & (pred != y)).sum())
    obj = {'n': len(y), 'accepted': accepted, 'errors': errors,
           'coverage': float(accept.mean()) if len(y) else None,
           'selective_error': errors/accepted if accepted else None,
           'worst_class_automatic_error': max(r['automatic_error'] for r in rows if r['n']),
           'newcastle_false_healthy': rows[3]['false_healthy']}
    for r in rows:
        for key in ['coverage', 'automatic_error', 'accepted', 'errors']:
            obj[r['class']+'_'+key] = r[key]
    return obj


class Context:
    def __init__(self, input_root, output_root, allow_missing_models=False):
        self.root = self.input_root = Path(input_root).resolve()
        self.out = self.output_root = Path(output_root).resolve()
        if self.root == self.out or self.root in self.out.parents:
            raise ValueError('Revision output must be outside frozen input directory')
        self.out.mkdir(parents=True, exist_ok=True)
        self.inputs = {}
        self.resume_omissions = []
        self.allow_missing_models = allow_missing_models
        self.project = Path(__file__).resolve().parents[1]
        self.cfg = self.read_json('experiment_freeze.json')['config']
        if self.cfg['classes'] != CLASSES:
            raise ValueError('Frozen class order mismatch')
        self.manifest = self.read_csv('data/manifest.csv')

    def track(self, path):
        path = Path(path)
        if not path.is_absolute():
            path = self.root/path
        path = path.resolve()
        try:
            key = path.relative_to(self.root).as_posix()
        except ValueError:
            try:
                key = 'revision:'+path.relative_to(self.out).as_posix()
            except ValueError:
                try:
                    artifact_revision = self.root.parent/'revision_ijies_20267101'
                    key = 'artifact-revision:'+path.relative_to(artifact_revision).as_posix()
                except ValueError:
                    try:
                        key = 'project:'+path.relative_to(self.project).as_posix()
                    except ValueError:
                        key = 'external:'+path.name
        self.inputs[key] = sha256(path)
        return path

    def read_json(self, path):
        return json.loads(self.track(path).read_text(encoding='utf-8'))

    def read_csv(self, path, **kwargs):
        return pd.read_csv(self.track(path), **kwargs)

    def predictions(self, run_path, fmt, split):
        path = self.track(Path(run_path)/f'predictions_{fmt}_{split}.npz')
        return load_predictions(path, self.manifest, split)

    def runs(self):
        for backbone in self.cfg['backbones']:
            for seed in self.cfg['seeds']:
                yield backbone, seed, self.root/backbone/f'seed_{seed}'

    def write_json(self, path, obj):
        path = Path(path)
        if not path.is_absolute():
            path = self.out/path
        if self.out != path and self.out not in path.parents:
            raise ValueError('Output escapes revision directory')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(clean_json(obj), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
        return path

    def table(self, name, rows):
        frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        path = self.out/'tables'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path.with_suffix('.csv'), index=False, lineterminator='\n', float_format='%.15g')
        labels = list(frame.columns)
        def cell(value):
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return 'null'
            if isinstance(value, float):
                return format(value, '.10g')
            return str(value).replace('|', '\\|').replace('\n', ' ')
        text = '# '+name+'\n\n'+EVIDENCE+'\n\n'
        text += '| '+' | '.join(labels)+' |\n| '+' | '.join(['---']*len(labels))+' |\n'
        text += ''.join('| '+' | '.join(cell(v) for v in row)+' |\n' for row in frame.itertuples(index=False, name=None))
        path.with_suffix('.md').write_text(text, encoding='utf-8')
        return frame

    def verify(self):
        freeze = self.read_json('experiment_freeze.json')
        spec = {k: freeze[k] for k in ['config', 'manifest_sha256', 'source_sha256']}
        fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        if fingerprint != freeze['fingerprint']:
            raise ValueError('Experiment-freeze fingerprint mismatch')
        for relative, expected in freeze['source_sha256'].items():
            actual = sha256(self.project/relative)
            if actual != expected:
                raise ValueError('Frozen source mismatch: '+relative)
            self.inputs['project:'+relative] = actual
        if sha256(self.root/'data/manifest.csv') != freeze['manifest_sha256']:
            raise ValueError('Frozen manifest hash mismatch')
        omitted = []
        statuses = []
        coverages = []
        for backbone, seed, run in self.runs():
            for freeze_name in ['selection_freeze.json', 'policy_freeze.json']:
                record = self.read_json(run/freeze_name)
                for name, expected in record['files'].items():
                    target = run/name
                    if not target.exists() and self.allow_missing_models and target.suffix == '.tflite':
                        export = self.read_json(run/f"export_{name.split('_')[1].split('.')[0]}.json")
                        if export['sha256'] != expected:
                            raise ValueError('Omitted model export metadata mismatch: '+name)
                        omitted.append((target.relative_to(self.root).as_posix(), expected))
                        continue
                    if sha256(self.track(target)) != expected:
                        raise ValueError('Selection/policy freeze mismatch: '+str(target))
            certs = self.read_json(run/'certificates.json')
            for fmt in self.cfg['formats']:
                test = self.predictions(run, fmt, 'test')
                if len(test['y']) != 1154:
                    raise ValueError('Primary test count mismatch')
                from poultryrisk.risk import apply_policy, cp_upper
                for rule in ['global', 'proposed']:
                    cert = certs[fmt][rule]
                    if cert['family_size'] != 24 or cert['delta_per_test'] != 0.05/96:
                        raise ValueError('Primary multiplicity allocation mismatch')
                    if [b['class'] for b in cert['class_bounds']] != CLASSES:
                        raise ValueError('Certificate class order mismatch')
                    statuses.append(cert['status'])
                    failed = [b['class'] for b in cert['class_bounds'] if not b['passes']]
                    if cert['status'] == 'CERTIFIED':
                        if backbone != 'mobilenetv2' or failed:
                            raise ValueError('Primary passing rule mismatch')
                        _, accept = apply_policy(test['p'], cert['deployed_policy'])
                        coverages.append(float(accept.mean()))
                    elif cert['status'] != 'UNCERTIFIED_FALLBACK_ALL_DEFER' or failed != ['newcastle']:
                        raise ValueError('Primary failure status mismatch')
                    for bound in cert['class_bounds']:
                        if not np.isclose(cp_upper(bound['errors'], bound['n'], 0.05/96), bound['upper_bound'], atol=1e-12):
                            raise ValueError('Primary numerical bound mismatch')
                        if bound['passes'] != (bound['upper_bound'] <= bound['alpha']):
                            raise ValueError('Primary bound pass indicator mismatch')
        if len(statuses) != 24 or statuses.count('CERTIFIED') != 7:
            raise ValueError('Primary 7/24 headline mismatch')
        if round(min(coverages)*100, 2) != 68.46 or round(max(coverages)*100, 2) != 76.52:
            raise ValueError('Primary coverage range mismatch')
        return {'source_files_verified': 8, 'run_freezes_verified': 12, 'primary_certified': 7,
                'primary_failed': 17, 'test_n': 1154, 'coverage_min': min(coverages),
                'coverage_max': max(coverages), 'omitted_models_metadata_verified': omitted,
                'scope': 'Native model hashes verified only when model files are present; no training or inference.'}

    def save_manifest(self, phase=None):
        path = self.out/'revision_manifest.json'
        previous = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        inputs = dict(previous.get('inputs', {}))
        inputs.update(self.inputs)
        # Live execution logs, private montages and document QA evolve outside
        # deterministic scientific phases and must not invalidate their reuse.
        excluded = {'logs', 'internal', 'document_qa', '__pycache__', 'manuscript_evidence.json'}
        outputs = {p.relative_to(self.out).as_posix(): sha256(p) for p in sorted(self.out.rglob('*'))
                   if p.is_file() and p != path and not excluded.intersection(p.relative_to(self.out).parts)
                   and '.tmp' not in p.suffixes}
        sources = {p.relative_to(self.project).as_posix(): sha256(p)
                   for pattern in ['revision/*.py', 'scripts/revision_*.py', 'tests/test_revision*.py']
                   for p in sorted(self.project.glob(pattern)) if p.is_file()}
        phases = dict(previous.get('phases', {}))
        if phase:
            phases[phase] = {'status': 'completed', 'scope': EVIDENCE}
        self.write_json(path, {'schema_version': 1, 'evidence': EVIDENCE, 'inputs': inputs,
                               'outputs': outputs, 'phases': phases,
                               'revision_source_sha256': sources,
                               'resume_omissions': self.resume_omissions,
                               'excluded_from_phase_reuse': sorted(excluded),
                               'path_semantics': 'plain input keys relative to frozen input; project: relative to checkout; revision: relative to current revision output; artifact-revision: relative to sibling historical revision artifacts; outputs relative to revision output'})

    def _saved_input_path(self, key, revision_root=None):
        """Resolve portable input keys, including manifests made before v2 keys."""
        revision_root = Path(revision_root) if revision_root else self.out
        if key.startswith('revision:'):
            return revision_root/key[len('revision:'):]
        if key.startswith('artifact-revision:'):
            return self.root.parent/'revision_ijies_20267101'/key[len('artifact-revision:'):]
        if key.startswith('project:'):
            relative = key[len('project:'):]
            # Older manifests recorded figure source tables as project inputs.
            prefix = 'results/revision_ijies_20267101/'
            if relative.startswith(prefix):
                suffix = relative[len(prefix):]
                current = revision_root/suffix
                return current if current.exists() else self.root.parent/'revision_ijies_20267101'/suffix
            return self.project/relative
        if key.startswith('external:'):
            raise ValueError('External input cannot be resolved portably on resume: '+key)
        return self.root/key

    def _omitted_input_evidence(self, key, expected, record):
        """Allow only declared artifact-only omissions, with explicit evidence."""
        if not self.allow_missing_models:
            return None
        if key == 'project:archive.zip':
            if expected != '6641a21b18310e23c1b848c99c0552b66eaf58bb3821f52563aecec448e0704e':
                raise ValueError('Private montage archive hash differs from declared original')
            return 'private montage archive intentionally omitted; no dataset image is needed for public scientific tables'
        if not key.startswith(('project:', 'revision:', 'artifact-revision:', 'external:')) and key.endswith('.tflite'):
            target = self.root/key
            export = target.with_name('export_'+target.stem.split('_')[1]+'.json')
            if export.exists() and self.read_json(export)['sha256'] == expected:
                return 'native model intentionally omitted; frozen export SHA256 matches; graph inspection uses hash-bound cache'
        prefix = 'project:results/session_20260923T013608010722Z/'
        if key.startswith(prefix):
            snapshot_key = 'project:revision/freeze_export_metadata.json'
            snapshot = self.project/'revision/freeze_export_metadata.json'
            snapshot_hash = record.get('inputs', {}).get(snapshot_key)
            if not snapshot_hash or not snapshot.is_file() or sha256(snapshot) != snapshot_hash:
                raise ValueError('Missing original stage log lacks its unchanged provenance snapshot')
            metadata = json.loads(snapshot.read_text(encoding='utf-8'))
            relative = key[len('project:results/'):]
            if any(row.get('source_relative_path') == relative and row.get('source_sha256') == expected
                   for row in metadata.get('stage_execution', {}).values()):
                return 'original execution log intentionally omitted; unchanged snapshot preserves its hash and reported chronology; original log bytes not rehashed'
        return None

    def validate_saved_inputs(self, record, revision_root=None):
        self.resume_omissions = []
        for key, expected in record.get('inputs', {}).items():
            path = self._saved_input_path(key, revision_root)
            if not path.is_file():
                evidence = self._omitted_input_evidence(key, expected, record)
                if evidence:
                    self.resume_omissions.append({'input': key, 'sha256': expected, 'evidence': evidence})
                    continue
                raise ValueError('Saved revision input missing: '+key)
            if sha256(path) != expected:
                raise ValueError('Saved revision input changed: '+key)

    def validate_existing_outputs(self):
        path = self.out/'revision_manifest.json'
        if not path.exists():
            # In an artifact-only checkout, the shipped reference manifest also
            # binds test predictions, which the pre-test freeze cannot include.
            reference = self.root.parent/'revision_ijies_20267101'/'revision_manifest.json'
            if self.allow_missing_models and reference.is_file() and reference.parent != self.out:
                record = json.loads(reference.read_text(encoding='utf-8'))
                self.validate_saved_inputs(record, revision_root=reference.parent)
                for relative, expected in record.get('outputs', {}).items():
                    artifact = reference.parent/relative
                    if not artifact.is_file() or sha256(artifact) != expected:
                        raise ValueError('Historical revision artifact changed or missing: '+relative)
            return {}
        record = json.loads(path.read_text(encoding='utf-8'))
        self.validate_saved_inputs(record)
        for relative, expected in record.get('outputs', {}).items():
            if not (self.out/relative).is_file() or sha256(self.out/relative) != expected:
                raise ValueError('Existing revision output changed: '+relative)
        return record.get('phases', {})
