''' An example of training a reinforcement learning agent on the environments in RLCard
'''
import os
import argparse
import sys
from pathlib import Path

# Make UNOFastApi the first path directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
#print(f"PROJECT ROOT: {PROJECT_ROOT}, SYS.PATH: {sys.path}")

import torch
import rlcard

import time
from math import floor

from backend.uno_agents import *
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)

def train(args):

    start = time.time()

    # Check whether gpu is available
    #device = get_device()
    device = torch.device("cpu") # cpu option is faster for MacBook air
        
    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(
        args.env,
        config={
            'seed': args.seed,
        }
    )

    # Initialize the agent
    layers = [256, 256]
    if args.algorithm == 'dqn':
        from rlcard.agents import DQNAgent
        if args.load_checkpoint_path != "":
            agent = DQNAgent.from_checkpoint(checkpoint=torch.load(args.load_checkpoint_path))
        else:
            agent = DQNAgent(
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                mlp_layers=layers,
                device=device,
                save_path=args.log_dir,
                save_every=args.save_every
            )

    elif args.algorithm == 'nfsp':
        from rlcard.agents import NFSPAgent
        if args.load_checkpoint_path != "":
            agent = NFSPAgent.from_checkpoint(checkpoint=torch.load(args.load_checkpoint_path))
        else:
            agent = NFSPAgent(
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                hidden_layers_sizes=[64,64],
                q_mlp_layers=[64,64],
                device=device,
                save_path=args.log_dir,
                save_every=args.save_every
            )
    agents = [agent]

    # Set opponent agent
    opponent_agent = RandomAgent()
    for _ in range(1, env.num_players):
        agents.append(opponent_agent)

    print(f"AGENTS: {agents}")
    env.set_agents(agents)

    # Start training
    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):
            if args.algorithm == 'nfsp':
                agents[0].sample_episode_policy()

            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Reorganize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)

            # Feed transitions into agent memory, and train the agent
            # Here, we assume that DQN always plays the first position
            # and the other players play randomly (if any)
            for ts in trajectories[0]:
                agent.feed(ts)

            # Evaluate the performance. Play against the selected heuristic agent
            if episode % args.evaluate_every == 0:
                logger.log_performance(
                    episode,
                    tournament(
                        env,
                        args.num_eval_games,
                    )[0]
                )

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    #plot_curve(csv_path, fig_path, args.algorithm)

    # Record the training configuration
    config_path = os.path.join(args.log_dir, "configuration.txt")
    with open(config_path, "w") as file:
        file.write(
            f"environment: {args.env}\n"
            f"algorithm: {args.algorithm}\n"
            f"opponent: {opponent_agent.name}\n"
            f"episodes: {args.num_episodes}\n"
            f"seed: {args.seed}\n"
            f"network: {layers}\n"
            f"evaluation games: {args.num_eval_games}\n"
            f"evaluation frequency: {args.evaluate_every}\n"
            f"save frequency: {args.save_every}\n"
            f"device: {device}\n"
            f"checkpoint loaded: {args.load_checkpoint_path if args.load_checkpoint_path else 'None'}\n"
        )
    print('Training configuration saved in', config_path)

    # Save model
    save_path = os.path.join(args.log_dir, f'{args.algorithm}_model.pth')
    torch.save(agent, save_path)
    print('Model saved in', save_path)

    end = time.time()
    training_time = end - start
    seconds = round(training_time % 60)
    minutes = floor(training_time / 60)
    print(f'Training time: {minutes} min {seconds} seconds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN/NFSP example in RLCard")
    parser.add_argument(
        '--env',
        type=str,
        default='uno',
        choices=[
            'blackjack',
            'leduc-holdem',
            'limit-holdem',
            'doudizhu',
            'mahjong',
            'no-limit-holdem',
            'uno',
            'gin-rummy',
            'bridge',
        ],
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        default='dqn',
        choices=[
            'dqn',
            'nfsp',
        ],
    )
    parser.add_argument(
        '--cuda',
        type=str,
        default='',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=5000,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/uno_dqn_result/',
    )
    
    parser.add_argument(
        "--load_checkpoint_path",
        type=str,
        default="",
    )
    
    parser.add_argument(
        "--save_every",
        type=int,
        default=1000)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)

