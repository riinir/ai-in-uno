import torch
from rlcard.agents.human_agents.uno_human_agent import HumanAgent
import os

from rlcard.agents import DQNAgent
from rlcard.utils import get_device
import rlcard as rlcard
import numpy

env = rlcard.make('uno')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# from backend.NewAIModule import suggest_move

global player_id, stt, trajectories, human_agent, ai_played_draw


def load_model(model_path, env=None, position=None, device=None):
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


def draw_card_backend():
    return env.returDrawnCardsFromEnv()


def is_draw_card_at_first(card):
    return "draw" in card




# Load the AI agent
def load_ai_agent():
    try:
        # TODO: #AIINGAMES change based on directory
        # r"C:\Users\abhar\extras\UNI\AI-IN-GAMES\AIinGames\UNOFastAPI\backend\model.pth"
        file_path = "/Users/ryannguyen/UnityProjects/ai-in-uno/UNOFastAPI/backend/model.pth"
        # Check if the file exists
        if os.path.exists(file_path):
            print("The file exists.")
        else:
            print("The file does not exist.")

        dqn_agent = load_model(file_path, env, device=get_device())
        print("AI agent loaded successfully.")
        return dqn_agent
    except Exception as e:
        print(f"Error loading AI agent: {e}")
        return None


# Set the agents in the environment
def set_agents(ai_agent):
    try:
        env.set_agents([human_agent, ai_agent])
        print("Agents set in the environment successfully.")
    except Exception as e:
        print(f"Error setting agents in the environment: {e}")


# Get the current game state for player 0
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


def run(action):
    # player_id = state['current_player']
    global trajectories, player_id, trajectories, ai_played_draw
    # if "draw" in action:



    state = get_game_state(player_id)

    trajectories[player_id].append(state)

    if not env.is_over():
        if player_id == 0:

            next_state, next_player_id = env.step(action, True)
        else:
            # next_state, next_player_id = env.step(action, env.agents[player_id].use_raw)
            ai_action = env.agents[1].step(env.get_state(1))  # Get AI's action (AI is player 1)
            ai_action_string = env.decode_action_api(ai_action)
            if ai_action_string == 'draw':
                ai_played_draw = True
            else:
                ai_played_draw = False
            next_state, next_player_id = env.step(ai_action)  # Apply the AI's action to the environment
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


def suggestion():
    ai_suggestion = env.agents[1].step(env.get_state(0))
    # UnoEnv.decode_action_api(ai_suggestion)

    suggested_action = env.decode_action_api(ai_suggestion)
    return suggested_action
