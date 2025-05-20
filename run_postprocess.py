#!/usr/bin/env python3

import os
import argparse
import numpy as np

from modules import QualControlDataIO
from modules import LabelExcInh
from modules import DffTraces

def read_ops(session_data_path):
    print(f'Processing {session_data_path}')
    ops = np.load(
        os.path.join(session_data_path, 'suite2p', 'plane0', 'ops.npy'),
        allow_pickle=True).item()
    ops['save_path0'] = session_data_path
    return ops

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Postprocess suite2p results.')

    parser.add_argument('--session_data_path', required=True, type=str,
                        help='Path to suite2p session folder.')
    parser.add_argument('--range_skew', nargs=2, type=float, required=True,
                        help='Two floats: skew range (e.g., -5 5)')
    parser.add_argument('--max_connect', type=float, required=True,
                        help='Maximum allowed connectivity.')
    parser.add_argument('--max_aspect', type=float, required=True,
                        help='Maximum allowed aspect ratio.')
    parser.add_argument('--range_footprint', nargs=2, type=float, required=True,
                        help='Two floats: footprint range (e.g., 1 2)')
    parser.add_argument('--range_compact', nargs=2, type=float, required=True,
                        help='Two floats: compactness range (e.g., 0 1.05)')
    parser.add_argument('--diameter', type=float, required=True,
                        help='Diameter for Cellpose segmentation.')

    args = parser.parse_args()

    # Read ops
    ops = read_ops(args.session_data_path)

    # Run quality control
    QualControlDataIO.run(
        ops,
        range_skew=np.array(args.range_skew),
        max_connect=args.max_connect,
        max_aspect=args.max_aspect,
        range_compact=np.array(args.range_compact),
        range_footprint=np.array(args.range_footprint)
    )

    # Run label extraction and dff computation
    LabelExcInh.run(ops, args.diameter)
    DffTraces.run(ops, correct_pmt=False)