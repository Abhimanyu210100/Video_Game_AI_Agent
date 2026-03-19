# Pokemon Red GenAI Agent (PyBoy)

Run Pokemon Red with a visible emulator window while a decision agent chooses inputs from the current screen.

## Requirements

- Python 3.10+
- A legal Pokemon Red ROM file (`.gb`)

## Setup

1. Activate your conda environment:
   - `conda activate video_game_agent`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Create your env file:
   - `cp .env.example .env`
4. Edit `.env` and set `ROM_PATH` to your ROM:
   - `ROM_PATH=/absolute/path/to/your/PokemonRed.gb`

## Run

Run with default settings:

`python train.py`

Run with visible fast-forward:

`python train.py --speed 3`

(`--speed` range is 1 to 6.)

Run with constant fallback action (useful smoke test):

`python train.py --fallback-action A`

## Notes

- The game window opens visibly via PyBoy (not headless).
- Replace `default_decision_fn(...)` in `train.py` with your GenAI API call.
- Your model should return exactly one action from: `A, B, START, SELECT, UP, DOWN, LEFT, RIGHT`.
