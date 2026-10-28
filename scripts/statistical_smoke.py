"""Exercise validation -> certification -> reporting with SYNTHETIC probabilities."""
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from poultryrisk.common import ROOT, load_config, read_json, write_json
from poultryrisk.pipeline import calibrate_run, load_predictions, select_run, verify_policy_freeze
from poultryrisk.evaluate import make_report

cfg = load_config(ROOT/'configs/pilot.yaml')
cfg['bootstrap_repetitions'] = 100
rng = np.random.default_rng(73)
with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp)
    rows = []
    for split in ['val', 'calibration', 'test']:
        y = np.repeat(np.arange(4), 200)
        ids = np.array([f'{split}_{i:05}' for i in range(len(y))])
        logits = rng.normal(0, 1, (len(y), 4))
        logits[np.arange(len(y)), y] += 4
        exp = np.exp(logits-logits.max(axis=1, keepdims=True))
        p = exp/exp.sum(axis=1, keepdims=True)
        rows.extend({'image_id': i, 'y': int(c), 'split': split} for i,c in zip(ids,y))
        for fmt in cfg['formats']:
            np.savez(out/f'predictions_{fmt}_{split}.npz', image_id=ids, y=y, p=p)
        if split == 'val':
            select_run(pd.DataFrame(rows), cfg, out)
    frame = pd.DataFrame(rows)
    calibrate_run(frame, cfg, out)
    verify_policy_freeze(out)
    for fmt in cfg['formats']:
        test = load_predictions(out/f'predictions_{fmt}_test.npz', frame, 'test')
        result = make_report(test['y'], test['p'], read_json(out/'policies.json')[fmt], out/f'eval_{fmt}', cfg)
        assert len(result['methods']) >= 6
    print('PASS: synthetic probability pipeline; no real poultry results generated; TensorFlow not exercised.')
