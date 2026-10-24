#!/usr/bin/env python3
"""Reproduce both passive experiments from source geometry and fixed parameters."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='New or empty results directory; existing results are not overwritten.')
    parser.add_argument('--harmonic-only', action='store_true',
                        help='Omit the 18 rectangular-current integrations; this is a partial reproduction.')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory is not empty; choose a new path.')
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[key] = '1'
    env['MPLBACKEND'] = 'Agg'
    env['MPLCONFIGDIR'] = str(output / '.matplotlib-cache')
    commands = [
        [sys.executable, 'reproduction/y_cable/run.py', '--output', str(output / 'y_cable')],
        [sys.executable, 'reproduction/allen/run.py', '--output', str(output / 'allen')]
        + ([] if args.harmonic_only else ['--include-transients']),
        [sys.executable, 'reproduction/verify_y_arithmetic.py',
         '--raw', str(output / 'y_cable/raw_data'),
         '--processed', str(output / 'y_cable/processed_data'),
         '--output', str(output / 'verification_y_arithmetic.json')],
        [sys.executable, 'reproduction/verify_allen_arithmetic.py',
         '--results', str(output / 'allen'),
         '--output', str(output / 'verification_allen_arithmetic.json')]
        + ([] if args.harmonic_only else ['--require-transients']),
        [sys.executable, 'reproduction/plot_results.py', '--results', str(output),
         '--output', str(output / 'figures')],
    ]
    ledger = []
    started = time.monotonic()
    for index, command in enumerate(commands, 1):
        print(f'[{index}/{len(commands)}] ' + ' '.join(command[1:]), flush=True)
        step_start = time.monotonic()
        with (output / f'command_{index}.stdout').open('w') as stdout, \
                (output / f'command_{index}.stderr').open('w') as stderr:
            result = subprocess.run(command, cwd=root, env=env, stdout=stdout, stderr=stderr)
        ledger.append({'argv': command, 'exit_code': result.returncode,
                       'wall_seconds': time.monotonic() - step_start})
        (output / 'commands.json').write_text(json.dumps(ledger, indent=2) + '\n')
        if result.returncode:
            print(f'Command failed; see command_{index}.stderr. Evidence retained at {output}', file=sys.stderr)
            return result.returncode
    y_summary = json.loads((output / 'y_cable/reproduction_status.json').read_text())
    allen_summary = json.loads((output / 'allen/summary.json').read_text())
    if (y_summary['reproduction_status'] != 'PASS'
            or y_summary['conditions'] != 35
            or y_summary['frequencies_per_condition'] != 101
            or y_summary['witnesses'] != 70
            or allen_summary['reproduction_status'] != 'PASS'
            or allen_summary['harmonic_conditions'] != 128
            or (not args.harmonic_only and (
                allen_summary['transient_runs'] != 18
                or allen_summary['transient_scientific_gate'] != 'FAIL'
                or not allen_summary['transient_failure_is_preserved']))):
        raise RuntimeError('Required experiment counts or reproduced verdicts are inconsistent.')
    files = {}
    for path in sorted(output.rglob('*')):
        if path.is_file() and '.matplotlib-cache' not in path.parts:
            files[str(path.relative_to(output))] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {
        'status': 'PASS',
        'meaning': 'The requested numerical reproduction commands and figure generation completed; '
                   'scientific acceptance criteria are reported separately by each experiment.',
        'scope': 'PARTIAL_HARMONIC_ONLY' if args.harmonic_only else 'FULL_PASSIVE_EXPERIMENTS',
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'wall_seconds': time.monotonic() - started,
        'python': sys.version,
        'experiment_summaries': {'y_cable': y_summary, 'allen': allen_summary},
        'transients_executed': not args.harmonic_only,
        'expected_scientific_failure': 'The reconstructed distal rectangular-current family fails '
                                       'the unchanged 0.0025 mV temporal accuracy criterion.',
        'commands': ledger,
        'output_sha256': files,
    }
    (output / 'REPRODUCTION.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('output_sha256', 'commands')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
