import sys
from pathlib import Path

# Adjust project path to /UNOFastAPI
# REMOVE .parent IF THIS FILE IS MOVED UP A DIRECTORY
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from fastapi import FastAPI
from ai_backend import *

# Initialize FastAPI app
app = FastAPI()

@app.get("/start_game")
def start_game():
    initialize_game()
    # Load AI agent and set up agents in the environment
    ai_agent = load_ai_agent("rlcard_rule")
    set_agents(ai_agent)

@app.get("/draw_card")
def draw_card():
    return draw_card_backend()

@app.get("/get_human_game_state")
def get_human_game_state_endpoint():
    """
    Returns the current state of the game for player 0.
    """
    return get_game_state()

@app.get("/get_ai_game_state")
def get_ai_game_state_endpoint():
    """
    Returns the current state of the game for player 0.
    """
    return get_game_state(1)


@app.post("/play_card")
async def play_card(card: str):
    """
    FastAPI endpoint for processing the AI's move.
    """
    # result = player_move(card_move)
    state_game = run(card)
    return state_game

@app.get("/suggestion")
async def suggest():
    """
    FastAPI endpoint for processing the AI's move.
    """
    # result = player_move(card_move)
    return suggestion()
#
# @app.get("/draw_card")
# def get_draw_card_endpoint():
#     """
#     Returns the current state of the game for player 0.
#     """
#     return "r-3"

# @app.get("/get_ai_move")
# def get_ai_move_endpoint():
#     """
#     Returns the current state of the game for player 0.
#     """
#     return "draw"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
