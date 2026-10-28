"""Finite-family, validation-selected policies and independent certification.

Risk R_c = P(accepted AND predicted != c | Y=c).
This is NOT error conditional on acceptance. Guarantees require independent,
identically distributed calibration samples within each class and future data.
"""
import math
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from scipy.stats import beta

from .common import probabilities


def cp_upper(k, n, delta):
    if n < 0 or not 0 <= k <= n or not 0 < delta < 1:
        raise ValueError('Invalid binomial bound arguments')
    if n == 0 or k == n:
        return 1.0
    return float(beta.ppf(1-delta, k+1, n-k))


def cp_interval(k, n, delta=0.05):
    if n == 0:
        return [None, None]
    return [0.0 if k == 0 else float(beta.ppf(delta/2, k, n-k+1)),
            1.0 if k == n else float(beta.ppf(1-delta/2, k+1, n-k))]


def minimum_zero_error_n(alpha, delta):
    return int(math.ceil(math.log(delta) / math.log1p(-alpha)))


def fit_temperature(p, y):
    p = probabilities(p)
    logp = np.log(np.clip(p, 1e-12, 1))
    result = minimize_scalar(lambda t: -np.log(np.clip(softmax(logp/t, axis=1)[np.arange(len(y)), y], 1e-12, 1)).mean(),
                             bounds=(0.1, 10), method='bounded')
    return float(result.x)


def transform(p, temperature=1.0):
    return softmax(np.log(np.clip(probabilities(p), 1e-12, 1))/temperature, axis=1)


def apply_policy(p, policy):
    p = transform(p, policy.get('temperature', 1.0))
    if policy['kind'] == 'conformal':
        sets = (1-p) <= np.asarray(policy['q'])[None, :]
        accept = sets.sum(axis=1) == 1
        return sets.argmax(axis=1), accept
    pred = p.argmax(axis=1)
    if policy.get('all_defer', False):
        return pred, np.zeros(len(p), dtype=bool)
    order = np.sort(p, axis=1)
    accepted = (p.max(axis=1) >= np.asarray(policy['thresholds'])[pred]) & ((order[:, -1]-order[:, -2]) >= policy['margin'])
    return pred, accepted


def class_risks(y, pred, accept):
    return np.array([float(np.mean((pred[y == c] != c) & accept[y == c])) if np.any(y == c) else 1.0 for c in range(4)])


def tune_policy(p, y, cfg, classwise=False, temperature=1.0):
    p = probabilities(p)
    q = transform(p, temperature)
    pred = q.argmax(axis=1)
    candidates = []
    if classwise:
        for level in cfg['quantile_grid']:
            thresholds = []
            for c in range(4):
                # Uses only validation examples; no calibration/test outcomes.
                scores = q[(pred == c) & (y == c), c]
                thresholds.append(float(np.quantile(scores, level)) if len(scores) else 1.01)
            for margin in cfg['margin_grid']:
                candidates.append({'kind': 'threshold', 'thresholds': thresholds, 'margin': margin, 'temperature': temperature})
    else:
        candidates = [{'kind': 'threshold', 'thresholds': [float(t)]*4, 'margin': 0.0, 'temperature': temperature}
                      for t in cfg['threshold_grid']]
    best = {'kind': 'threshold', 'thresholds': [1.01]*4, 'margin': 0.0,
            'temperature': temperature, 'all_defer': True}
    best_coverage = -1
    for policy in candidates:
        pred, accept = apply_policy(p, policy)
        risk = class_risks(y, pred, accept)
        if np.all(risk <= np.asarray(cfg['alpha']) * cfg['tuning_safety_fraction']):
            coverage = float(accept.mean())
            if coverage > best_coverage:
                best, best_coverage = policy, coverage
    return best


def certify(p, y, policy, cfg, family_size):
    if family_size < 1:
        raise ValueError('family_size must be positive')
    pred, accept = apply_policy(p, policy)
    d = cfg['delta'] / (family_size * 4)
    rows = []
    for c in range(4):
        mask = y == c
        n = int(mask.sum())
        k = int(((pred != y) & accept & mask).sum())
        bound = cp_upper(k, n, d)
        rows.append({'class': cfg['classes'][c], 'n': n, 'errors': k, 'upper_bound': bound,
                     'alpha': cfg['alpha'][c], 'passes': bound <= cfg['alpha'][c],
                     'minimum_n_if_zero_errors': minimum_zero_error_n(cfg['alpha'][c], d)})
    structural = policy.get('all_defer', False) or min(policy['thresholds']) > 1
    passed = all(x['passes'] for x in rows)
    deployed = dict(policy)
    status = 'CERTIFIED' if passed else 'UNCERTIFIED_FALLBACK_ALL_DEFER'
    if structural:
        status = 'TRIVIAL_ALL_DEFER'
        deployed['all_defer'] = True
    elif not passed:
        deployed['all_defer'] = True
    return {'status': status, 'selected_policy': policy, 'deployed_policy': deployed,
            'class_bounds': rows, 'delta_per_test': d, 'family_size': family_size,
            'risk_definition': 'P(automatic incorrect prediction | true class)',
            'assumption': 'iid within true class; fixed policies selected without calibration labels; no arbitrary-shift guarantee'}


def conformal_policy(p, y, alpha, temperature=1.0):
    p = transform(p, temperature)
    quantiles = []
    for c in range(4):
        scores = np.sort(1-p[y == c, c])
        k = math.ceil((len(scores)+1)*(1-alpha[c]))
        quantiles.append(float(scores[k-1]) if 1 <= k <= len(scores) else 1.0)
    return {'kind': 'conformal', 'q': quantiles, 'temperature': temperature,
            'guarantee_type': 'class-conditional marginal set coverage under exchangeability; not a high-probability conditional-risk certificate'}
