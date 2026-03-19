"""Gymnasium environment wrapper around PyBoy for Pokemon Red."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from dotenv import load_dotenv
from gymnasium import spaces
from pyboy import PyBoy
from pyboy.utils import WindowEvent

class PokemonRedEnv(gym.Env):
    """Gymnasium-compatible PyBoy environment with visible rendering."""

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, speed: int = 1):
        super().__init__()
        load_dotenv()

        rom_path = os.getenv("ROM_PATH")
        if not rom_path:
            default_rom = (Path(__file__).resolve().parent / "../pokered/pokered.gbc").resolve()
            if default_rom.exists():
                rom_path = str(default_rom)
                print(f"ROM_PATH not set; using default ROM at: {rom_path}")
            else:
                raise FileNotFoundError(
                    "ROM_PATH is not set. Create a .env file from .env.example and set ROM_PATH."
                )

        self.rom_path = Path(rom_path).expanduser().resolve()
        if not self.rom_path.exists():
            raise FileNotFoundError(
                f"ROM file not found at '{self.rom_path}'. Update ROM_PATH in .env."
            )

        self.state_path = self.rom_path.parent / "initial.state"
        self.speed = int(np.clip(speed, 1, 6))
        self.pyboy: Optional[PyBoy] = None

        # 8 actions: A, B, Start, Select, Up, Down, Left, Right
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
        )

        self._action_press = {
            0: WindowEvent.PRESS_BUTTON_A,
            1: WindowEvent.PRESS_BUTTON_B,
            2: WindowEvent.PRESS_BUTTON_START,
            3: WindowEvent.PRESS_BUTTON_SELECT,
            4: WindowEvent.PRESS_ARROW_UP,
            5: WindowEvent.PRESS_ARROW_DOWN,
            6: WindowEvent.PRESS_ARROW_LEFT,
            7: WindowEvent.PRESS_ARROW_RIGHT,
        }
        self._action_release = {
            0: WindowEvent.RELEASE_BUTTON_A,
            1: WindowEvent.RELEASE_BUTTON_B,
            2: WindowEvent.RELEASE_BUTTON_START,
            3: WindowEvent.RELEASE_BUTTON_SELECT,
            4: WindowEvent.RELEASE_ARROW_UP,
            5: WindowEvent.RELEASE_ARROW_DOWN,
            6: WindowEvent.RELEASE_ARROW_LEFT,
            7: WindowEvent.RELEASE_ARROW_RIGHT,
        }

        self._init_emulator()

    def _init_emulator(self) -> None:
        """Start PyBoy with a visible window and initialize save state."""
        if self.pyboy is not None:
            self.pyboy.stop(save=False)

        # Keep window visible: no window_type='headless'
        self.pyboy = PyBoy(str(self.rom_path), window_type="SDL2")
        self.pyboy.set_emulation_speed(self.speed)

        if not self.state_path.exists():
            # Give the game time to boot before capturing baseline state.
            for _ in range(120):
                self.pyboy.tick()
            with open(self.state_path, "wb") as state_file:
                self.pyboy.save_state(state_file)

        self._load_initial_state()

    def _load_initial_state(self) -> None:
        if self.pyboy is None:
            raise RuntimeError("PyBoy is not initialized.")
        with open(self.state_path, "rb") as state_file:
            self.pyboy.load_state(state_file)
        for _ in range(5):
            self.pyboy.tick()

    def _get_observation(self) -> np.ndarray:
        if self.pyboy is None:
            raise RuntimeError("PyBoy is not initialized.")

        frame = np.asarray(self.pyboy.screen.ndarray, dtype=np.uint8)
        # frame shape expected (144, 160, 4) RGBA on most backends
        if frame.ndim == 3 and frame.shape[-1] >= 3:
            frame = frame[..., :3]
        obs = self._resize_nn(frame, 84, 84)
        return obs

    @staticmethod
    def _resize_nn(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        """Lightweight nearest-neighbor resize without extra dependencies."""
        in_h, in_w = image.shape[:2]
        y_idx = np.linspace(0, in_h - 1, out_h).astype(np.int32)
        x_idx = np.linspace(0, in_w - 1, out_w).astype(np.int32)
        return image[y_idx][:, x_idx]

    def _press_action(self, action: int) -> None:
        if self.pyboy is None:
            raise RuntimeError("PyBoy is not initialized.")
        self.pyboy.send_input(self._action_press[action])
        self.pyboy.tick()
        self.pyboy.send_input(self._action_release[action])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self.pyboy is None:
            raise RuntimeError("PyBoy is not initialized.")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        self._press_action(int(action))
        for _ in range(24):
            self.pyboy.tick()

        reward = 0.0
        obs = self._get_observation()

        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._load_initial_state()
        obs = self._get_observation()
        return obs, {}

    def render(self):
        # PyBoy handles rendering in its own visible window.
        return None

    def close(self) -> None:
        if self.pyboy is not None:
            self.pyboy.stop(save=False)
            self.pyboy = None
