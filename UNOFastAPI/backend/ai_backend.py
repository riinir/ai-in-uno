import torch
from rlcard.agents.human_agents.uno_human_agent import HumanAgent
import os
from pathlib import Path

import rlcard as rlcard
from rlcard.agents import DQNAgent
from rlcard.utils import get_device
from rlcard import models as rlcard_models
from uno_agents import *
import numpy

# Create rlcard UNO environment
env = rlcard.make('uno')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

global player_id, stt, trajectories, human_agent, ai_played_draw

AGENT_REGISTRY = {
    "random": RandomAgent,
    "aggressive": AggressiveAgent,
    "conservative": ConservativeAgent,
    "balanced": BalancedAgent,
}

def load_ai_agent(agent_type="dqn"):
    try:
        if agent_type == "dqn":
            # Get the absolute path to /model.pth
            model_path = Path(__file__).resolve().parent / "uno_agents" / "model.pth"
            # Check if the file exists
            # if os.path.exists(model_path):
            #    print("The file exists.")
            # else:
            #    print("The file does not exist.")
            agent = load_model(model_path, env, device=get_device())
        elif agent_type == "rlcard_rule":
            agent = rlcard_models.load("uno-rule-v1").agents[0]
        # Custom (non-rlcard) agent types
        elif agent_type in AGENT_REGISTRY:
            agent = AGENT_REGISTRY[agent_type]()

        print(f"{agent_type} agent loaded successfully.")
        return agent
    except Exception as e:
        print(f"Error loading {agent_type} agent: {e}")
        return None

def load_model(model_path, device=None):
    print("hi")
    if os.path.isfile(model_path):
        print("Loading model from {}".format(model_path))
        # Torch model
        import torch
        torch.serialization.safe_globals([numpy.core.multiarray._reconstruct])
        torch.serialization.add_safe_globals([DQNAgent])
        agent = torch.load(model_path, map_location=device, weights_only=False)
        agent.set_device(device)
    # elif os.path.isdir(model_path):  # CFR model
    #     from rlcard.agents import CFRAgent
    #     agent = CFRAgent(env, model_path)
    #     agent.load()
    # elif model_path == 'random':  # Random model
    #     from rlcard.agents import RandomAgent
    #     agent = RandomAgent(num_actions=env.num_actions)
    # else:  # A model in the model zoo
    #     from rlcard import models
    #     agent = models.load(model_path).agents[position]

    return agent

def set_agents(ai_agent):
    try:
        env.set_agents([human_agent, ai_agent])
        print("Agents set in the environment successfully.")
    except Exception as e:
        print(f"Error setting agents in the environment: {e}")


def initialize_game():
    # Initialize the environment and agents
    # env.reset()
    global stt, player_id, trajectories, human_agent, ai_played_draw
    # stt, player_id = env.reset()
    # last_played = stt.get("played_cards", [])[0] if stt.get("played_cards") else None
    while True:
        stt, player_id = env.reset()
        played_cards = env.get_state(player_id)['raw_obs'].get("played_cards", [])

        if played_cards and ("draw" in played_cards[-1] or "wild" in played_cards[-1] or
                             "skip" in played_cards[-1] or "reverse" in played_cards[-1]):
            # A "draw" card was played — reset again
            continue
        else:
            # No "draw" card — break the loop and proceed
            break
    # while True:
    #     # stt, player_id = env.reset()
    #
    #     # Example: assuming `state` is your dictionary like the one you posted
    #     last_played = get_game_state(0)
    #     # last_played = get_game_state(0)
    #     # print(last_played)
    #     if last_played and is_draw_card_at_first(last_played):
    #         continue  # restart loop, i.e., reset again
    #     else:
    #         break  # exit the loop, valid star
    # while not is_draw_card_at_first():
    #     stt, player_id = env.reset()

    # the first player is always human
    # player_id = 0

    # Create the human agent
    human_agent = HumanAgent(env.num_actions)
    trajectories = [[] for _ in range(env.num_players)]
    ai_played_draw = False

    # print(get_game_state(0))

def run(action):
    # player_id = state['current_player']
    global trajectories, player_id, trajectories, ai_played_draw
    # if "draw" in action:

    state = get_game_state(player_id)

    trajectories[player_id].append(state)

    if not env.is_over():
        # Human player's action
        if player_id == 0:
            next_state, next_player_id = env.step(action, True)
        # AI player's action
        else:
            ai_action = env.agents[1].step(env.get_state(1))  # Get AI's action (AI is player 1)

            # DQN agent uses encoded actions rather than raw action strings, so we need to check to adjust accordingly
            if env.agents[1].use_raw:
                ai_action_string = ai_action
            else:
                # Decode the DQN agent's action
                ai_action_string = env.decode_action_api(ai_action)

            if ai_action_string == 'draw':
                ai_played_draw = True
            else:
                ai_played_draw = False

            # Apply the AI's action to the environment
            next_state, next_player_id = env.step(ai_action, env.agents[1].use_raw)
            # state = get_game_state()  # Get the new game state

        trajectories[player_id].append(action)

        # Set the state and player
        # state = next_state
        player_id = next_player_id

        # Save state.
        # if not env.game.is_over():
        #     trajectories[player_id].append(state)
        # for pid in range(env.num_players):
        #     state = env.get_state(pid)
        #     trajectories[pid].append(state)
        #
        # # Payoffs
        payoffs = env.get_payoffs()
        state = get_game_state(player_id)
        return state

    return None

def suggestion():
    ai_suggestion = env.agents[1].step(env.get_state(0))

    # Some agents use the raw action string (e.g., "r-7"), so we can return that immediately
    if env.agents[1].use_raw:
        return ai_suggestion

    # Else decode it first from integer to action string then return
    return env.decode_action_api(ai_suggestion)

def draw_card_backend():
    return env.returDrawnCardsFromEnv()

def is_draw_card_at_first(card):
    return "draw" in card


# Get the current game state for player 0 (Human player)
def get_game_state(player_id=0):
    global ai_played_draw
    state_info = env.get_state(player_id)  # Get the state for player 0
    raw_state = state_info['raw_obs']

    game_state = {
        'hand': raw_state.get('hand', []),  # Player's hand
        'target': raw_state.get('target', None),  # Target card on the pile
        'played_cards': raw_state.get('played_cards', []),  # Cards played
        'legal_actions': raw_state.get('legal_actions', []),  # Legal actions available
        'num_cards': raw_state.get('num_cards', []),  # Number of cards per player
        'num_players': raw_state.get('num_players', 2),  # Number of players
        'current_player': raw_state.get('current_player', 0),  # Current player
        'ai_played_draw': ai_played_draw
    }
    return game_state
