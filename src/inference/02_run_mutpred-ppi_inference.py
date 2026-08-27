'''
run_model_inference_pipeline_scaledstability_finetune.py
Author: Ross Stewart
Date: November 2025
runs MutPred-PPI inference on a preformatted dataset and saves results

Interaction prediction labels:
- 1: Disrupted interaction (variant disrupts protein-protein interaction)
- 0: Unperturbed interaction (variant maintains wild-type interaction)

Usage:
    python 02_run_mutpred-ppi_inference.py <working_dir> --device <device> [--models-dir <path>]
'''

import sys
import argparse
import os
from utils.inference_utils import run_inference_on_dataset
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

_DEFAULT_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../', 'models_new'))

# parse arguments
parser = argparse.ArgumentParser(description='Run MutPred-PPI inference')
parser.add_argument('working_dir', help='Working directory containing af3_graphs/ and wt_and_vt.fasta')
parser.add_argument('--device', default='cuda:0', help='GPU device (default: cuda:0)')
parser.add_argument('--models-dir', default=_DEFAULT_MODELS_DIR,
                    help='Directory containing model .pt files and mutation_diff_scaler.pkl '
                         '(default: models_new/ next to publication root, loads _all.pt for inference)')
args = parser.parse_args()

device = args.device
working_dir = args.working_dir
models_dir = args.models_dir

graph_dir = f'{working_dir}/af3_graphs'
t5_fasta_path = f'{working_dir}/wt_and_vt.fasta'
results_dir= f'{working_dir}/results'
os.makedirs(results_dir, exist_ok=True)

if not os.path.exists(graph_dir) or not os.path.exists(t5_fasta_path):
    print(f'Error: {graph_dir} or {t5_fasta_path} does not exist')
    sys.exit(1)

run_inference_on_dataset(device, working_dir, graph_dir, t5_fasta_path, results_dir, models_dir=models_dir)
