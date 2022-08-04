import argparse
from distutils.util import strtobool
from encodings import normalize_encoding
from locale import normalize
import runpy
import time
import numpy as np
import sys
import copy
import wandb


if __name__ == "__main__":
    run_name = f"tuner_{int(time.time())}"
    wandb.init(
        project="cleanRL",
        name=run_name,
        monitor_gym=True,
        save_code=True,
    )
    target_scores = {
        "pong_v3": None,
        # "double_dunk_v3": None,
        # "ice_hockey_v2": None,
        # "tennis_v3": None,
        # "boxing_v2": None,
        # "surround_v2": None,
    }
    print('ARGV:', sys.argv)
    total_runs = 4
    concurrent_runs = 2
    num_seeds = total_runs // concurrent_runs  # may be remainder

    main_args = copy.deepcopy(sys.argv)
    scores = []
    for env in target_scores.keys():
        for seed in range(num_seeds):
            sys.argv = main_args + ["--env-id", env, "--seed", str(seed)]
            ppo = runpy.run_path(path_name='cleanrl/ppo_pettingzoo_ma_atari.py', run_name="__main__")
            print(f"The average episodic return on {env} is {np.average(ppo['avg_returns'])}")
            scores += [max(0, np.average(ppo['avg_returns']))]
            # normalized_scores += [max(0, np.average(ppo['avg_returns']) / target_scores[env])]

    print(f"The scores are {np.average(scores)}")
    wandb.log({"scores": np.average(scores)})