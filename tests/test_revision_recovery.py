"""Regression checks for fail-closed, relocatable saved-artifact recovery."""
import json
from pathlib import Path
import tempfile
import unittest
from revision.common import Context, sha256
from revision.phase_a import operator_table_rows


class RevisionRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        base = Path(self.temp.name)
        self.ctx = Context.__new__(Context)
        self.ctx.project = base/'package'
        self.ctx.root = self.ctx.project/'artifacts/full_colab_v1'
        self.ctx.out = self.ctx.project/'results/reproduced_revision'
        self.ctx.inputs = {}
        self.ctx.resume_omissions = []
        self.ctx.allow_missing_models = False
        self.ctx.root.mkdir(parents=True)
        self.ctx.out.mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def put(self, path, content='saved evidence'):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        return sha256(path)

    def test_changed_test_prediction_bytes_are_rejected(self):
        path = self.ctx.root/'mobilenetv2/seed_17/predictions_fp32_test.npz'
        expected = self.put(path)
        self.ctx.validate_saved_inputs({'inputs': {'mobilenetv2/seed_17/predictions_fp32_test.npz': expected}})
        self.put(path, 'same IDs could still hide changed probabilities')
        with self.assertRaisesRegex(ValueError, 'input changed'):
            self.ctx.validate_saved_inputs({'inputs': {'mobilenetv2/seed_17/predictions_fp32_test.npz': expected}})

    def test_legacy_revision_table_key_relocates(self):
        path = self.ctx.out/'tables/A1.csv'
        expected = self.put(path)
        self.ctx.validate_saved_inputs({'inputs': {'project:results/revision_ijies_20267101/tables/A1.csv': expected}})
        self.assertEqual(self.ctx.track(path), path)
        self.assertIn('revision:tables/A1.csv', self.ctx.inputs)

    def test_changed_operator_cache_is_rejected_after_relocation(self):
        path = self.ctx.root.parent/'revision_ijies_20267101/inspection/operator_inspection.json'
        expected = self.put(path, '{}')
        record = {'inputs': {'artifact-revision:inspection/operator_inspection.json': expected}}
        self.ctx.validate_saved_inputs(record)
        self.put(path, '{"changed": true}')
        with self.assertRaisesRegex(ValueError, 'input changed'):
            self.ctx.validate_saved_inputs(record)

    def test_archive_omission_requires_artifact_only_mode_and_declared_hash(self):
        record = {'inputs': {'project:archive.zip': '6641a21b18310e23c1b848c99c0552b66eaf58bb3821f52563aecec448e0704e'}}
        with self.assertRaisesRegex(ValueError, 'input missing'):
            self.ctx.validate_saved_inputs(record)
        self.ctx.allow_missing_models = True
        self.ctx.validate_saved_inputs(record)
        self.assertEqual(len(self.ctx.resume_omissions), 1)
        with self.assertRaisesRegex(ValueError, 'hash differs'):
            self.ctx.validate_saved_inputs({'inputs': {'project:archive.zip': 'invalid'}})

    def test_stage_log_omission_needs_unchanged_hash_bound_snapshot(self):
        relative = 'session_20260923T013608010722Z/full_evaluate.json'
        snapshot = self.ctx.project/'revision/freeze_export_metadata.json'
        obj = {'stage_execution': {'test_predictions': {'source_relative_path': relative, 'source_sha256': 'original-log-digest'}}}
        digest = self.put(snapshot, json.dumps(obj))
        record = {'inputs': {'project:results/'+relative: 'original-log-digest',
                             'project:revision/freeze_export_metadata.json': digest}}
        self.ctx.allow_missing_models = True
        self.ctx.validate_saved_inputs(record)
        self.assertEqual(len(self.ctx.resume_omissions), 1)
        self.put(snapshot, '{}')
        with self.assertRaisesRegex(ValueError, 'unchanged provenance snapshot'):
            self.ctx.validate_saved_inputs(record)

    def test_fresh_artifact_reproduction_checks_reference_test_hashes(self):
        reference = self.ctx.root.parent/'revision_ijies_20267101'
        test = self.ctx.root/'predictions_test.npz'
        expected = self.put(test)
        self.put(reference/'revision_manifest.json', json.dumps({'inputs': {'predictions_test.npz': expected}}))
        self.ctx.allow_missing_models = True
        self.assertEqual(self.ctx.validate_existing_outputs(), {})
        self.put(test, 'tampered numerical predictions')
        with self.assertRaisesRegex(ValueError, 'input changed'):
            self.ctx.validate_existing_outputs()

    def test_fresh_artifact_reproduction_checks_published_inspection_cache(self):
        reference = self.ctx.root.parent/'revision_ijies_20267101'
        cache = reference/'inspection/operator_inspection.json'
        expected = self.put(cache, '{"models": []}')
        self.put(reference/'revision_manifest.json', json.dumps({'outputs': {'inspection/operator_inspection.json': expected}}))
        self.ctx.allow_missing_models = True
        self.assertEqual(self.ctx.validate_existing_outputs(), {})
        self.put(cache, '{"models": [{"changed_counts": true}]}')
        with self.assertRaisesRegex(ValueError, 'Historical revision artifact changed'):
            self.ctx.validate_existing_outputs()

    def test_native_and_json_cached_operator_tables_have_identical_bytes(self):
        native = {'models': [{'backbone': 'toy', 'seed': 17, 'format': 'int8',
                              'status': 'INSPECTED', 'operators': 1, 'float_tensor_operators': 0}],
                  'operators': [{'backbone': 'toy', 'seed': 17, 'format': 'int8',
                                 'operator_index': 0, 'operator_name': 'ADD', 'uses_float_tensor': False}]}
        cached = json.loads(json.dumps(native, sort_keys=True))
        cold_models, cold_operators = operator_table_rows(native)
        warm_models, warm_operators = operator_table_rows(cached)
        self.ctx.table('cold_models', cold_models)
        self.ctx.table('warm_models', warm_models)
        self.ctx.table('cold_operators', cold_operators)
        self.ctx.table('warm_operators', warm_operators)
        for suffix in ['models', 'operators']:
            self.assertEqual(sha256(self.ctx.out/'tables'/('cold_'+suffix+'.csv')),
                             sha256(self.ctx.out/'tables'/('warm_'+suffix+'.csv')))


if __name__ == '__main__':
    unittest.main()
