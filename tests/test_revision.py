"""Synthetic statistical checks for post-hoc procedures, not evidence of i.i.d."""
import unittest

import numpy as np
from scipy.stats import binom

from revision.statistics import (aurc, confidence_scores, crc_threshold,
                                 fixed_sequence_from_counts, k_max, ltt_threshold)


class RevisionStatisticsTests(unittest.TestCase):
    def test_aurc_manual_discrete_example_and_oracle(self):
        result = aurc([0, 1, 0], [3., 2., 1.])
        self.assertAlmostEqual(result['aurc'], (0+1/2+1/3)/3)
        self.assertAlmostEqual(result['oracle_aurc'], 1/9)
        self.assertAlmostEqual(result['e_aurc'], 1/6)

    def test_aurc_tie_expectation_is_permutation_invariant(self):
        a = aurc([0, 1, 0], [1., 1., 1.])
        b = aurc([1, 0, 0], [1., 1., 1.])
        self.assertEqual(a, b)
        self.assertAlmostEqual(a['aurc'], 1/3)

    def test_iut_single_rule_exact_size_and_simulation(self):
        # One truly unsafe class suffices to make the union null true.
        n, alpha, delta, true_bad_risk = 84, .1, .05, .10001
        critical = k_max(n, delta, alpha)
        exact_rejection_probability = binom.cdf(critical, n, true_bad_risk)
        self.assertLessEqual(exact_rejection_probability, delta)
        rng = np.random.default_rng(20261027)
        counts = rng.binomial(n, [.005, .005, .005, true_bad_risk], size=(10000, 4))
        rejected = (binom.cdf(counts, n, alpha).max(axis=1) <= delta)
        self.assertLessEqual(rejected.mean(), delta)

    def test_iut_24_rule_fwer_exact_and_simulation(self):
        n, m, delta, alpha = 84, 24, .05, .1
        critical = k_max(n, delta/m, alpha)
        individual_size = binom.cdf(critical, n, .10001)
        self.assertLessEqual(m*individual_size, delta)
        rng = np.random.default_rng(20261028)
        counts = rng.binomial(n, [.002, .002, .002, .10001], size=(6000, m, 4))
        false_rejections = binom.cdf(counts, n, alpha).max(axis=2) <= delta/m
        self.assertLessEqual(false_rejections.any(axis=1).mean(), delta)

    def test_ltt_fixed_sequence_stops_at_first_nonrejection(self):
        last, ps = fixed_sequence_from_counts([[0]*4, [1]*4, [0]*4], [84]*4, delta=.05/96)
        self.assertEqual(last, 0)
        self.assertLessEqual(ps[0], .05/96)
        self.assertGreater(ps[1], .05/96)

    def test_ltt_iut_fwer_simulation(self):
        rng = np.random.default_rng(20261029)
        n, trials, delta = 150, 2500, .05
        y = np.repeat(np.arange(4), n)
        grid = np.array([.9, .8, .7, .6, .5, .4, .3, .2, .1, 0.])
        probabilities = np.repeat([.01, .01, .01, .2], n)
        false_certifications = 0
        for _ in range(trials):
            score = rng.random(len(y))
            incorrect = rng.random(len(y)) < probabilities
            pred = np.where(incorrect, (y+1) % 4, y)
            threshold, index, _ = ltt_threshold(y, pred, score, grid, delta=delta)
            if index >= 0 and .2*(1-threshold) > .1+1e-12:
                false_certifications += 1
        self.assertLessEqual(false_certifications/trials, delta)

    def test_crc_classwise_expected_risk_simulation(self):
        # Every prediction is wrong, so population risk at t is 1-t for U[0,1].
        # This exposes the finite-sample +1 correction directly.
        rng = np.random.default_rng(20261030)
        n, trials, alpha = 99, 4000, .1
        y = np.repeat(np.arange(4), n)
        pred = (y+1) % 4
        individual, combined = [], []
        for _ in range(trials):
            threshold, per_class = crc_threshold(y, pred, rng.random(len(y)), alpha)
            individual.append(1-np.asarray(per_class))
            combined.append(1-threshold)
        means = np.mean(individual, axis=0)
        # Monte Carlo variation is explicitly allowed; exact order-statistic
        # expectation for each class is 10/(99+1)=0.10.
        self.assertTrue(np.all(np.abs(means-alpha) < .003))
        self.assertLessEqual(np.mean(combined), alpha)

    def test_crc_excludes_complete_score_tie(self):
        y = np.repeat(np.arange(4), 99)
        threshold, _ = crc_threshold(y, (y+1) % 4, np.full(len(y), .5))
        self.assertGreater(threshold, .5)
        self.assertFalse(np.any(np.full(len(y), .5) >= threshold))

    def test_bootstrap_paired_deterministic_and_identity(self):
        from revision.phase_b import paired_bootstrap
        y = np.repeat(np.arange(4), 8)
        pred = y.copy(); pred[::5] = (pred[::5]+1) % 4
        a = np.arange(len(y)) % 3 != 0
        b = np.arange(len(y)) % 4 != 0
        first = paired_bootstrap(y, pred, a, pred, b, repetitions=100, seed=17)
        self.assertEqual(first, paired_bootstrap(y, pred, a, pred, b, repetitions=100, seed=17))
        same = paired_bootstrap(y, pred, a, pred, a, repetitions=100, seed=17)
        for row in same:
            for key in ('estimate', 'ci_low', 'ci_high'):
                self.assertEqual(row[key], 0.)

    def test_score_orientations_and_energy(self):
        p = np.array([[.7, .1, .1, .1], [.4, .2, .2, .2]])
        result = confidence_scores(p, np.log(p))
        self.assertGreater(result['doctor'][0], result['doctor'][1])
        self.assertGreater(result['negative_entropy'][0], result['negative_entropy'][1])
        np.testing.assert_allclose(result['energy'], [0., 0.], atol=1e-15)


if __name__ == '__main__':
    unittest.main()
