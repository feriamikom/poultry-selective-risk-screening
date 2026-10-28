import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False, default=str), encoding='utf-8')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    assert cfg['classes'] == ['healthy', 'coccidiosis', 'salmonella', 'newcastle']
    assert abs(sum(cfg['splits'].values()) - 1) < 1e-9
    assert set(cfg['splits']) == {'train', 'val', 'calibration', 'test'}
    assert all(0 < x < 1 for x in cfg['splits'].values())
    assert len(cfg['alpha']) == 4 and all(0 < x < 1 for x in cfg['alpha'])
    assert 0 < cfg['delta'] < 1
    assert len(set(cfg['formats'])) == len(cfg['formats'])
    assert set(cfg['formats']).issubset({'fp32', 'int8'}) and 'fp32' in cfg['formats']
    assert len(set(cfg['seeds'])) == len(cfg['seeds'])
    assert len(set(cfg['backbones'])) == len(cfg['backbones'])
    assert set(cfg['backbones']).issubset({'mobilenetv2', 'efficientnetb0', 'tinycnn'})
    return cfg


def resolve_path(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def probabilities(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 4 or len(x) == 0:
        raise ValueError('Expected nonempty N x 4 probabilities')
    if not np.isfinite(x).all() or (x < 0).any() or (x > 1.00001).any():
        raise ValueError('Invalid probability values')
    if not np.allclose(x.sum(axis=1), 1, atol=1e-4):
        raise ValueError('Probabilities must sum to one')
    return x
