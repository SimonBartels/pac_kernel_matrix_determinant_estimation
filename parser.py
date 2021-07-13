import argparse
import numpy as np

from registry import KERNEL_DICT, ALGORITHM_DICT, CLIP
from result_management import ALGORITHM


def get_default_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", "--dataset", type=str, default="zeros_1000")
    parser.add_argument("-sn2", "--sn2", type=float, default=1e-3)
    parser.add_argument("-c", "--clip", type=float, default=CLIP)
    parser.add_argument("-k", "--kernel", type=str, choices=KERNEL_DICT.keys(), default=list(KERNEL_DICT.keys())[0])
    parser.add_argument("-k-ls", "--kernel-ls", type=float, default=0.)
    parser.add_argument("-k-var", "--kernel-var", type=float, default=0.)

    parser.add_argument("-v", "--verbose", type=bool, default=False)

    parser.add_argument("-en", "--experiment-name", type=str, default=None)

    parser.add_argument("-s", "--seed", type=int, default=0)

    parser.add_argument("-mi", "--max-iterations", type=int, default=201)
    parser.add_argument("-mt", "--max-time", type=float, default=np.infty)
    parser.add_argument("-a", "--" + ALGORITHM, type=str, choices=ALGORITHM_DICT.keys())

    for alg in ALGORITHM_DICT.values():
        alg.add_parameters(parser)

    return parser
