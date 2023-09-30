import argparse
import os
import subprocess
from tqdm import tqdm
from datetime import datetime
import json
import numpy as np

# Manual settings
coin_count = 50
agent_count = 1
file_out = "results/eval_metrics.jsonl"

# Automatic settings
now = datetime.now()
now_f = now.strftime("%Y-%m-%d_%H-%M-%S").split(".")[0]


def run():

    parser = argparse.ArgumentParser(description="Run current agent and export evaluation metrics")

    parser.add_argument("--match-name", "-n", dest="match_name", required="True", type=str, help="Name to give the current evaluation execution.")
    parser.add_argument("--qtable", "-q", required=True, type=str)
    parser.add_argument("--agent", "-a", default="lord_of_the_peanuts_agent", type=str, help="Agent found in ./agent_code/")
    parser.add_argument("--n-rounds", dest="n_rounds", default=100, type=int)

    args = parser.parse_args()

    agent = args.agent
    qtable = args.qtable
    n_rounds = args.n_rounds
    match_name = now_f + '_' + args.match_name

    command = f"python3 main.py play --match-name {match_name} --n-rounds {n_rounds} --agents {agent} --scenario coin-heaven --no-gui --save-stats --qtable {qtable}"
    print(command)
  
    subprocess.run(command, shell="True")
    print("done")

    with open(f"results/{match_name}.json", "r") as f:
        data = json.load(f)

    invalid_moves = data['by_agent'][agent].get('invalid', 0) / n_rounds

    coins = []
    kills = []
    suicides = []

    for round in data['by_round'].keys():

            coins.append(data['by_round'][round]['coins'])
            kills.append(data['by_round'][round]['kills'])
            suicides.append(data['by_round'][round]['suicides'])


    coins_mean = np.mean(coins)
    coins_std = np.std(coins)

    kills_mean = np.mean(kills)
    kills_std = np.std(kills)

    suicides_mean = np.mean(suicides)
    suicides_std = np.std(suicides)

    if os.path.isfile("results/win.log"):

        with open("results/win.log", "r") as f:

            lines = f.read().splitlines()
            lines = [int(i) for i in lines]

            steps_collect_coins_mean = np.mean(lines)
            steps_collect_coins_std = np.std(lines)

        os.remove("results/win.log")

    else:

        steps_collect_coins_mean = "NaN"
        steps_collect_coins_std = "NaN"

    eval_out = {
            'match_name': match_name,
            'invalid_moves': invalid_moves,
            'coins_mean': coins_mean,
            'coins_std': coins_std,
            'kills_mean': kills_mean,
            'kills_std': kills_std,
            'suicides_mean': suicides_mean,
            'suicides_std': suicides_std,
            'steps_collect_coins_mean': steps_collect_coins_mean,
            'steps_collect_coins_std': steps_collect_coins_std
            }

    with open(file_out, "a") as f:

        line = json.dumps(eval_out)
        f.write(line + '\n')

if __name__ == "__main__":
    run()
