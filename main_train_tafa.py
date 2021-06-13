import argparse
from rl_algorithms.agent import Agent
from rl_algorithms.utils import SaveJson
from rl_algorithms.rl_config import RLConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="PendulumAug-v0")
    parser.add_argument("--gpu_id", default=0)
    parser.add_argument("--seed", default=1, type=int)
    args = parser.parse_args()

    hyper_params = RLConfig(args.env, gpu_id=args.gpu_id, seed=args.seed, hyper_file_name=f'./exp_configs/{args.env}')
    json_saver = SaveJson()
    json_saver.save_file(f'{hyper_params.file_name}', hyper_params)
    agent = Agent(hyper_params)
    agent.training_process()


if __name__ == '__main__':
    main()









