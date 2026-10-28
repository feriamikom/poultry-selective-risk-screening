"""Reproduce post-hoc revision analyses without training or model inference."""
import argparse
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from revision.common import Context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=ROOT/'results/full_colab_v1')
    parser.add_argument('--output-root', type=Path, default=ROOT/'results/revision_ijies_20267101')
    parser.add_argument('--phase', choices=['verify', 'A', 'B', 'C', 'D', 'all'], default='all')
    parser.add_argument('--allow-missing-models', action='store_true', help='Artifact-only reproduction: verify omitted model metadata; reuse hash-bound operator-inspection cache')
    parser.add_argument('--force', action='store_true', help='Recompute requested phase; preserves immutable primary artifacts')
    args = parser.parse_args()
    ctx = Context(args.input_root, args.output_root, args.allow_missing_models)
    phases = ctx.validate_existing_outputs()
    integrity = ctx.verify()
    ctx.write_json('integrity_verification.json', integrity)
    print('Integrity passed: primary 7/24 unchanged; all source and available freeze hashes match.', flush=True)
    if args.phase == 'verify':
        ctx.save_manifest()
        return
    requested = ['A', 'B', 'C', 'D'] if args.phase == 'all' else [args.phase]
    for phase in requested:
        if phases.get(phase, {}).get('status') == 'completed' and not args.force:
            print('Reusing completed phase '+phase+'; output hashes verified.', flush=True)
            continue
        module = importlib.import_module('revision.phase_'+phase.lower())
        module.run(ctx)
        ctx.save_manifest(phase)
        print('Completed phase '+phase, flush=True)
    ctx.save_manifest()


if __name__ == '__main__':
    main()
