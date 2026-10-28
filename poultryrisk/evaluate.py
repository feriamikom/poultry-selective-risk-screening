import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

from .common import probabilities, write_json
from .risk import apply_policy, cp_interval


def metrics(y, p, policy):
    probabilities(p)
    pred, accepted = apply_policy(p, policy)
    wrong = accepted & (pred != y)
    n_accepted = int(accepted.sum())
    result = {'n': len(y), 'n_accepted': n_accepted, 'automatic_coverage': float(accepted.mean()),
              'selective_error': float(wrong.sum()/n_accepted) if n_accepted else None,
              'selective_error_ci95': cp_interval(int(wrong.sum()), n_accepted),
              'classwise': []}
    for c in range(4):
        mask = y == c
        n = int(mask.sum())
        errors = int((mask & wrong).sum())
        accepted_c = int((mask & accepted).sum())
        false_healthy = int((mask & accepted & (pred == 0)).sum()) if c else None
        result['classwise'].append({'class_id': c, 'n': n, 'n_accepted': accepted_c,
            'automatic_error': errors/n if n else None, 'automatic_error_ci95': cp_interval(errors, n),
            'coverage': accepted_c/n if n else None,
            'correct_automatic_fraction': int((mask & accepted & (pred == c)).sum())/n if n else None,
            'false_healthy_fraction': false_healthy/n if c and n else None})
    return result


def bootstrap_difference(y, p, a, b, repetitions, seed):
    # Pair identical images, stratify by true class. One retained image/group.
    rng = np.random.default_rng(seed)
    _, aa = apply_policy(p, a)
    _, ab = apply_policy(p, b)
    diff = aa.astype(float)-ab.astype(float)
    strata = [np.flatnonzero(y == c) for c in range(4)]
    values = []
    for _ in range(repetitions):
        ix = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata if len(s)])
        values.append(diff[ix].mean())
    return {'coverage_difference': float(diff.mean()), 'ci95': np.quantile(values, [0.025, 0.975]).tolist(),
            'repetitions': repetitions, 'unit': 'retained image group',
            'scope': 'descriptive paired interval; no farm-level independence established'}


def make_report(y, p, policies, out, cfg):
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    flat = []
    for name, policy in policies.items():
        result = metrics(y, p, policy)
        results[name] = result
        row = {k: v for k, v in result.items() if k != 'classwise' and not isinstance(v, list)}
        row['method'] = name
        for c in result['classwise']:
            label = cfg['classes'][c['class_id']]
            for k in ['automatic_error', 'coverage', 'false_healthy_fraction']:
                row[label+'_'+k] = c[k]
        flat.append(row)
    rawpred = p.argmax(axis=1)
    report = {'classification_baseline': {'accuracy': float((rawpred == y).mean()),
         'macro_f1': float(f1_score(y, rawpred, labels=range(4), average='macro', zero_division=0)),
         'recall_by_class': recall_score(y, rawpred, labels=range(4), average=None, zero_division=0).tolist()},
         'methods': results,
         'notes': ['Deferred images remain in classwise risk denominators.',
                   'Selective error is undefined (null), not zero, if all images are deferred.',
                   'Confidence intervals are descriptive; certification uses separate calibration data.']}
    if 'proposed_certified' in policies and 'global_certified' in policies:
        report['paired_coverage_comparison'] = bootstrap_difference(y, p, policies['proposed_certified'],
            policies['global_certified'], cfg['bootstrap_repetitions'], cfg['split_seed'])
    write_json(out / 'metrics.json', report)
    pd.DataFrame(flat).to_csv(out / 'summary.csv', index=False)
    return report
