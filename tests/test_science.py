import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from poultryrisk.common import ROOT, load_config, read_json
from poultryrisk.data import audit, filename_info, validate_manifest
from poultryrisk.evaluate import metrics
from poultryrisk.pipeline import (calibrate_run, load_predictions, select_run,
                                  verify_policy_freeze, verify_selection_freeze)
from poultryrisk.risk import (apply_policy, certify, conformal_policy, cp_upper,
                             minimum_zero_error_n)


def config():
    return load_config(ROOT/'configs/pilot.yaml')


def probability_fixture(n_per_class, seed=1):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(4), n_per_class)
    p = rng.uniform(0.005, 0.025, (len(y), 4))
    p[np.arange(len(y)), y] += 0.92
    return y, p/p.sum(axis=1, keepdims=True)


class StatisticalTests(unittest.TestCase):
    def test_zero_errors_does_not_prove_zero_risk(self):
        self.assertAlmostEqual(cp_upper(0, 20, 0.05), 1-0.05**(1/20))
        self.assertGreater(cp_upper(0, 20, 0.05), 0.10)
        self.assertEqual(cp_upper(0, 0, 0.05), 1)
        self.assertEqual(cp_upper(10, 10, 0.05), 1)

    def test_multiple_testing_is_more_conservative(self):
        self.assertGreater(cp_upper(0, 84, 0.05/16), cp_upper(0, 84, 0.05))
        self.assertGreater(minimum_zero_error_n(0.05, 0.05/16), 84)

    def test_full_family_and_zero_error_requirements(self):
        cfg = load_config(ROOT/'configs/full.yaml')
        family = 2*len(cfg['formats'])*len(cfg['backbones'])*len(cfg['seeds'])
        self.assertEqual(family, 24)
        self.assertEqual(minimum_zero_error_n(.1, cfg['delta']/(4*family)), 72)
        self.assertEqual(minimum_zero_error_n(.05, cfg['delta']/(4*family)), 148)

    def test_abstention_denominators(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([[.95,.02,.02,.01], [.94,.03,.02,.01], [.96,.01,.02,.01], [.25,.25,.25,.25]])
        policy = {'kind': 'threshold', 'thresholds': [.9]*4, 'margin': 0}
        result = metrics(y, p, policy)
        self.assertEqual(result['automatic_coverage'], 0.75)
        self.assertAlmostEqual(result['selective_error'], 1/3)
        self.assertEqual(result['classwise'][1]['automatic_error'], 0.5)
        self.assertEqual(result['classwise'][1]['false_healthy_fraction'], 0.5)
        policy['all_defer'] = True
        self.assertIsNone(metrics(y, p, policy)['selective_error'])

    def test_missing_class_cannot_be_certified(self):
        y, p = probability_fixture(2)
        mask = y != 3
        policy = {'kind': 'threshold', 'thresholds': [0]*4, 'margin': 0}
        cert = certify(p[mask], y[mask], policy, config(), 4)
        self.assertEqual(cert['status'], 'UNCERTIFIED_FALLBACK_ALL_DEFER')
        self.assertTrue(cert['deployed_policy']['all_defer'])

    def test_enough_clean_calibration_can_certify(self):
        y, p = probability_fixture(500)
        policy = {'kind': 'threshold', 'thresholds': [.9]*4, 'margin': 0}
        cert = certify(p, y, policy, config(), 4)
        self.assertEqual(cert['status'], 'CERTIFIED')

    def test_empty_conformal_class_included_conservatively(self):
        y, p = probability_fixture(20)
        policy = conformal_policy(p[y != 3], y[y != 3], [.05]*4)
        self.assertEqual(policy['q'][3], 1)
        # With no calibration examples in any class, every set includes all classes.
        empty_policy = {'kind': 'conformal', 'q': [1]*4, 'temperature': 1}
        _, accepted = apply_policy(p, empty_policy)
        self.assertFalse(accepted.any())

    def test_class_conditional_conformal_monte_carlo(self):
        # Independent repeated calibration/future draws, not hand-picked success examples.
        rng = np.random.default_rng(44)
        covered = []
        for _ in range(200):
            y = np.repeat(np.arange(4), 49)
            p = rng.dirichlet(np.ones(4), len(y))
            policy = conformal_policy(p, y, [.1]*4)
            future = rng.dirichlet(np.ones(4), 400)
            fy = np.repeat(np.arange(4), 100)
            covered.append(np.mean(1-future[np.arange(400), fy] <= np.array(policy['q'])[fy]))
        self.assertGreater(np.mean(covered), 0.88)


class IntegrityTests(unittest.TestCase):
    def test_filename_sources_are_only_proxies(self):
        self.assertEqual(filename_info('Train/pcrncd.7.jpg'), ('newcastle', 'pcr_prefix'))
        self.assertEqual(filename_info('Train/cocci.0.jpg'), ('coccidiosis', 'field_prefix'))

    def test_duplicate_conflicts_quarantined_and_splits_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root/'raw'
            data.mkdir()
            rng = np.random.default_rng(19)
            for name in ['healthy', 'cocci', 'salmo', 'ncd']:
                for i in range(16):
                    arr = rng.integers(0, 256, (40, 40, 3), dtype='uint8')
                    Image.fromarray(arr).save(data/f'{name}.{i}.png')
            # Cross-label duplicate pair is quarantined in its entirety.
            (data/'cocci.conflict.png').write_bytes((data/'healthy.0.png').read_bytes())
            (data/'salmo.copy.png').write_bytes((data/'salmo.0.png').read_bytes())
            cfg = config()
            cfg.update(data_dir=str(data), output_dir=str(root/'out'), phash_distance=0)
            f = audit(cfg)
            validate_manifest(cfg, verify_bytes=True)
            self.assertEqual(len(f), 63)
            self.assertFalse(f.group_id.duplicated().any())
            self.assertNotIn('healthy.0.png', set(f.path))
            self.assertEqual(set(f.split), {'train','val','calibration','test'})
            manifest = root/'out/data/manifest.csv'
            manifest.write_text(manifest.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'Manifest changed'):
                validate_manifest(cfg)

    def test_prediction_id_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'pred.npz'
            np.savez(path, image_id=np.array(['wrong']), y=np.array([0]), p=np.array([[1,0,0,0]]))
            frame = pd.DataFrame({'image_id':['right'], 'split':['val'], 'y':[0]})
            with self.assertRaises(ValueError):
                load_predictions(path, frame, 'val')

    def test_policy_selection_freeze_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            cfg = config()
            rows = []
            for split in ['val', 'calibration']:
                y, p = probability_fixture(100)
                ids = np.array([f'{split}_{i:05}' for i in range(len(y))])
                rows.extend({'image_id': i, 'y': int(c), 'split': split} for i,c in zip(ids,y))
                for fmt in cfg['formats']:
                    np.savez(out/f'predictions_{fmt}_{split}.npz', image_id=ids, y=y, p=p)
                if split == 'val':
                    select_run(pd.DataFrame(rows), cfg, out)
            frame = pd.DataFrame(rows)
            with patch('poultryrisk.pipeline.tune_policy', side_effect=AssertionError('No calibration retuning')):
                with patch('poultryrisk.pipeline.fit_temperature', side_effect=AssertionError('No calibration refit')):
                    calibrate_run(frame, cfg, out)
            self.assertFalse(read_json(out/'policy_freeze.json')['test_used'])
            verify_policy_freeze(out)
            (out/'policies.json').write_text('{}')
            with self.assertRaises(ValueError):
                verify_policy_freeze(out)

    def test_selection_cannot_be_backdated_after_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out/'predictions_fp32_calibration.npz').write_bytes(b'already created')
            with self.assertRaisesRegex(ValueError, 'already exist'):
                select_run(pd.DataFrame(), config(), out)

    def test_selection_frozen_before_calibration_and_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            cfg = config()
            y, p = probability_fixture(20)
            ids = np.array([f'val_{i:05}' for i in range(len(y))])
            frame = pd.DataFrame({'image_id': ids, 'y': y, 'split': 'val'})
            for fmt in cfg['formats']:
                np.savez(out/f'predictions_{fmt}_val.npz', image_id=ids, y=y, p=p)
            select_run(frame, cfg, out)
            self.assertFalse(read_json(out/'selection_freeze.json')['calibration_used'])
            verify_selection_freeze(out)
            (out/'selected_policies.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'selection inputs changed'):
                verify_selection_freeze(out)

    def test_inference_holdout_guards_run_before_tensorflow(self):
        from poultryrisk.models import infer
        with tempfile.TemporaryDirectory() as tmp:
            for split in ['calibration', 'test']:
                with self.assertRaises(FileNotFoundError):
                    infer(pd.DataFrame(), config(), Path(tmp), 'fp32', split)


if __name__ == '__main__':
    unittest.main()
