import argparse
import subprocess
from tqdm import tqdm
from datetime import datetime
import json

# Manual settings
coin_count = 9
agent_count = 3
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

    command = f"python3 main.py play --match-name {match_name} --n-rounds {n_rounds} --agents {agent} peaceful_agent peaceful_agent coin_collector_agent --scenario classic --no-gui --save-stats --qtable {qtable}"
    print(command)
  
    subprocess.run(command, shell="True")
    print("done")

    with open(f"results/{match_name}.json", "r") as f:
        data = json.load(f)

    if agent in ['peaceful_agent', 'random_agent', 'coin_collector_agent', 'rule_based_agent']:
        agent = list(data['by_agent'].keys())[0]

    invalid_moves = data['by_agent'][agent].get('invalid', 0) / n_rounds
    coins_mean = data['by_agent'][agent].get('coins', 0) / n_rounds
    kills_mean = data['by_agent'][agent].get('kills', 0) / n_rounds
    suicides_mean = data['by_agent'][agent].get('suicides', 0) / n_rounds
    steps_mean = data['by_agent'][agent].get('steps', 0) / n_rounds
    score_mean = data['by_agent'][agent].get('score', 0) / n_rounds

    eval_out = {
            'match_name': match_name,
            'invalid_moves': invalid_moves,
            'coins_mean': coins_mean,
            'kills_mean': kills_mean,
            'suicides_mean': suicides_mean,
            'steps_mean': steps_mean,
            'score_mean': score_mean
            }

    with open(file_out, "a") as f:

        line = json.dumps(eval_out)
        f.write(line + '\n')

if __name__ == "__main__":
    run()
