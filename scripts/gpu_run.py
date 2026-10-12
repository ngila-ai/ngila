"""Run a command on a currently idle GPU without changing other workloads."""
import argparse
import os
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True, type=int)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('A command is required')
    result = subprocess.check_output([
        'nvidia-smi', f'--id={args.gpu}', '--query-gpu=memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    memory, utilization = map(int, result.strip().split(','))
    if memory > 256 or utilization > 5:
        raise RuntimeError(f'GPU {args.gpu} is busy: {memory} MiB, {utilization}%')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
    raise SystemExit(subprocess.call(command, env=env))


if __name__ == '__main__':
    main()
