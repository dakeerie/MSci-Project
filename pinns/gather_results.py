import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=int, required=True)
parser.add_argument('--delta', type=float, required=True)
parser.add_argument('--n_runs', type=int, default=50)
args = parser.parse_args()

base = f'./RepeatRunQNM/l{args.mode}/delta{args.delta}'
output_file = os.path.join(base, f'qnm_results_l{args.mode}_delta{args.delta}_{args.n_runs}runs.txt')

rows = []
missing = []

for run_id in range(args.n_runs):
    result_path = os.path.join(base, str(run_id), 'result.txt')
    if not os.path.exists(result_path):
        missing.append(run_id)
        continue
    entry = {}
    with open(result_path, 'r') as f:
        for line in f:
            key, val = line.strip().split('=')
            entry[key] = float(val)
    rows.append([run_id, entry['initial_rho_re'], entry['initial_rho_im'],
                 entry['omega_re'], entry['omega_im'],
                 entry['rho_re'], entry['rho_im'], entry['final_distance']])

if missing:
    print(f"WARNING: Missing results for run IDs: {missing}")

rows = np.array(rows)
header = (f"l={args.mode}, delta={args.delta}, "
          f"n_completed={len(rows)}/{args.n_runs}\n"
          f"run_id  initial_rho_re  initial_rho_im  omega_re  omega_im  rho_re  rho_im  final_distance")
np.savetxt(output_file, rows, fmt=['%d', '%.8f', '%.8f', '%.8f', '%.8f', '%.8f', '%.8f', '%.8e'],
           header=header)
print(f"Results saved to {output_file}")