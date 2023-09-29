import argparse
import subprocess
from tqdm import tqdm
from datetime import datetime
import json
import numpy as np

# Manual settings
seed = 13
coin_count = 50
agent_count = 1
file_out = "results/eval_metrics.jsonl"

# Automatic settings
now = datetime.now()
now_f = now.strftime("%Y-%m-%d_%H-%M-%S").split(".")[0]


def metric_1(coins, 
               kills,
               suicides,
               n_rounds,
               coin_count,
               agent_count,
               w1=0.5,
               w2=0.5,
               w3=1):

    for w in [w1, w2, w3]:
        assert w <= 1
        assert w >= 0
    
    metric = w1 * np.mean(coins) / coin_count \
             + w2 * np.mean(kills) / agent_count \
             - w3 * np.mean(suicides) \
                * (w1 * np.mean(coins) / coin_count  + w1 * np.mean(coins) / coin_count)

    return metric.round(2)


def run():

    parser = argparse.ArgumentParser(description="Run current agent and export evaluation metrics")

    parser.add_argument("--match-name", "-n", dest="match_name", required="True", type=str, help="Name to give the current evaluation execution.")
    parser.add_argument("--qtable", "-q", required=True, type=str)
    parser.add_argument("--agent", "-a", default="lord_of_the_peanuts_agent", type=str, help="Agent found in ./agent_code/")
    parser.add_argument("--n-rounds", dest="n_rounds", default=1000, type=int)

    args = parser.parse_args()

    agent = args.agent
    qtable = args.qtable
    n_rounds = args.n_rounds
    match_name = now_f + '_' + args.match_name

    command = f"python main.py play --match-name {match_name} --n-rounds {n_rounds} --agents {agent} --seed {seed} --scenario loot-crate --no-gui --save-stats --qtable {qtable}"
    print(command)
  
    subprocess.run(command, shell="True")
    print("done")

    with open(f"results/{match_name}.json", "r") as f:
        data = json.load(f)

    invalid_moves = data['by_agent'][agent]['invalid'] / (n_rounds)

    coins = []
    kills = []
    suicides = []
    steps = []

    for round in data['by_round'].keys():

            coins.append(data['by_round'][round]['coins'])
            kills.append(data['by_round'][round]['kills'])
            suicides.append(data['by_round'][round]['suicides'])
            steps.append(data['by_round'][round]['steps'])


    coins_mean = np.mean(coins)
    coins_std = np.std(coins)

    kills_mean = np.mean(kills)
    kills_std = np.std(kills)

    suicides_mean = np.mean(suicides)
    suicides_std = np.std(suicides)

    steps_mean = np.mean(steps)
    steps_std = np.std(steps)

    metric1 = metric_1(coins=coins, 
                      kills=kills,
                      suicides=suicides,
                      n_rounds=n_rounds,
                      coin_count=coin_count,
                      agent_count=agent_count,
                      w1=1,
                      w2=0,
                      w3=0)


    eval_out = {
            'match_name': match_name,
            'invalid_moves': invalid_moves,
            'coins_mean': coins_mean,
            'coins_std': coins_std,
            'kills_mean': kills_mean,
            'kills_std': kills_std,
            'suicides_mean': suicides_mean,
            'suicides_std': suicides_std,
            'steps_mean': steps_mean,
            'steps_std': steps_std,
            'metric1': metric1,
            }

    with open(file_out, "a") as f:

        line = json.dumps(eval_out)
        f.write(line + '\n')

if __name__ == "__main__":
    run()
