import argparse
from rl_algorithms.agent import Agent
from rl_algorithms.rl_config import RLConfig
from rl_algorithms.show_attention_fig import show_env_attention


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="PendulumAug-v0")
    parser.add_argument("--num_checkpoint", default=300000, type=int)
    parser.add_argument("--gpu_id", default=0)
    parser.add_argument("--seed", default=1, type=int)
    args = parser.parse_args()

    hyper_params = RLConfig(args.env, gpu_id=args.gpu_id, seed=args.seed, hyper_file_name=f'./exp_configs/{args.env}')
    hyper_params.max_buffer_size = 1000
    agent = Agent(hyper_params)
    agent.load(f"./models/{hyper_params.exp_name}", args.num_checkpoint)
    agent.eval_process(eval_episodes=1, cur_step=args.num_checkpoint)
    show_env_attention(args.env, args.seed, num_checkpoint=args.num_checkpoint)


if __name__ == '__main__':
    main()









