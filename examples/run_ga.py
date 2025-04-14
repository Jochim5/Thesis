import random
import numpy as np
import argparse

from ga.run import run_ga
from ga.run_hand import run_one_ga
from ppo.args import add_ppo_args
from comparison.run import run_compare_ga


if __name__ == "__main__":
    seed = 9
    random.seed(seed)
    np.random.seed(seed)

    parser = argparse.ArgumentParser(description='Arguments for ga script')
    parser.add_argument('--exp-name', type=str, default='test_ga', help='Name of the experiment (default: test_ga)')
    parser.add_argument('--env-name', type=str, default='Walker-v0', help='Name of the environment (default: Walker-v0)')
    parser.add_argument('--pop-size', type=int, default=3, help='Population size (default: 3)')
    parser.add_argument('--structure_shape', type=tuple, default=(5, 5), help='Shape of the structure (default: (5,5))')
    parser.add_argument('--max-evaluations', type=int, default=6,
                    help='Maximum number of robots that will be evaluated (default: 6)')
    parser.add_argument('--num-cores', type=int, default=3, help='Number of robots to evaluate simultaneously (default: 3)')
    add_ppo_args(parser)
    parser.add_argument("--mode", type=str, default="original", help="Choose from: original, one, fast")
    parser.add_argument("--robot_path", type=str, default="C:\\d_pan\\PythonProject\\pythonProject\\pythonProject\\evogym-design-tool-main\\evogym-design-tool-main\\src\\exported\\speed_bot.json", help="Path to template robot JSON")
    parser.add_argument("--template_path", type=str, default="C:\\d_pan\\PythonProject\\pythonProject\\pythonProject\\evogym-design-tool-main\\evogym-design-tool-main\\src\\exported\\robot_Q.json", help="Path to template robot JSON")
    
    args = parser.parse_args()

    if args.mode == "original":
        run_ga(args)
    elif args.mode == "one":
        run_one_ga(args)
    elif args.mode == "fast":
        run_compare_ga(args)
