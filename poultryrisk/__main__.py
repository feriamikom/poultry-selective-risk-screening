import argparse
from .common import load_config, resolve_path, write_json
from .data import audit, extract_zip, validate_manifest
from .pipeline import doctor, run
from .risk import minimum_zero_error_n


def main():
    parser = argparse.ArgumentParser(description='Nafi-BIMA risk-control research starter')
    parser.add_argument('command', choices=['doctor', 'prepare', 'verify-data', 'feasibility', 'run'])
    parser.add_argument('--config', default='configs/pilot.yaml')
    parser.add_argument('--zip', dest='archive')
    parser.add_argument('--stage', choices=['train', 'export', 'select-policy', 'predict-calibration', 'calibrate', 'evaluate', 'all'], default='all')
    args = parser.parse_args()
    if args.command == 'doctor':
        doctor()
        return
    cfg = load_config(resolve_path(args.config))
    if args.command == 'prepare':
        if args.archive:
            extract_zip(resolve_path(args.archive), resolve_path(cfg['data_dir']))
        frame = audit(cfg)
        print(frame.groupby(['split', 'label']).size().unstack(fill_value=0).to_string())
    elif args.command == 'verify-data':
        frame = validate_manifest(cfg, verify_bytes=True)
        print(f'Verified {len(frame)} retained images and split integrity.')
    elif args.command == 'feasibility':
        frame = validate_manifest(cfg)
        family = 2*len(cfg['formats'])*len(cfg['backbones'])*len(cfg['seeds'])
        delta = cfg['delta']/(4*family)
        rows = []
        for c, name in enumerate(cfg['classes']):
            n = int(((frame.split == 'calibration') & (frame.y == c)).sum())
            for alpha in sorted(set([0.05, 0.10, cfg['alpha'][c]])):
                required = minimum_zero_error_n(alpha, delta)
                rows.append(dict(label=name, alpha=alpha, calibration_n=n, minimum_if_zero_errors=required,
                                 can_certify_with_zero_errors=n >= required))
        write_json(resolve_path(cfg['output_dir'])/'feasibility.json', rows)
        for row in rows:
            print(row)
        print('Feasibility is conditional on iid assumptions and zero errors; it is NOT a certificate.')
    elif args.command == 'run':
        run(cfg, args.stage)


if __name__ == '__main__':
    main()
