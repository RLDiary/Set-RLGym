"""
env.py
======
A high-quality SET card game environment with Cairo-powered graphics.

Overview:
---------
This module provides a complete SET game implementation featuring:
- Initial 12-card board guaranteed to contain at least one valid Set
- Dynamic board layout: 3 cards per column (12 cards = 4×3, 15 cards = 5×3, etc.)
- High-quality Cairo graphics with anti-aliasing and enhanced visibility
- Card numbering from 1-12 (or higher) instead of programmer 0-indexing
- Image-only I/O - agents interact purely through visual board representations
Indexing note: image overlays are 1-based; API indices are 0-based

Core Gameplay:
--------------
- select(i, j, k): Provide three board indices (0-based).
    • UI labels on the image are 1-based; convert label n → n-1.
    • Valid Set: Cards are removed, board refilled, returns success message + new image
    • Invalid Set: Returns error message + unchanged board image  
- deal_three(): Deals 3 additional cards (up to 21 max), returns status + new image
- reset(): Starts new game with fresh shuffled deck

Visual Features:
----------------
- Cairo-rendered shapes with 50% larger symbols for improved visibility
- Optimal spacing prevents shape overlap (especially for 3-shape cards)  
- Semi-transparent index pills with 24pt font for clear card identification
- Dynamic grid layout maintains 3 cards per column regardless of total count
- Procedural drawing of all SET elements (diamonds, squiggles, ovals; solid, striped, empty)

Card Encoding:
--------------
- Base-3 encoding [0..80] represents all 81 unique SET cards
- Feature order: NUMBER, COLOR, SHADING, SHAPE
- Each feature has 3 variants (0,1,2) following SET rules
- Index display uses human-friendly 1-based numbering

Public API:
-----------
class SetEnv:
    __init__(seed: int | None = None, require_initial_set: bool = True)
    reset(seed: int | None = None) -> Image
    get_board_image() -> Image  
    select(indices: tuple[int, int, int]) -> EnvReturn  # indices are 0-based positions
    deal_three() -> EnvReturn

class EnvReturn(NamedTuple):
    message: str | None  # Status/error message
    image: Image | None  # Updated board image

Dependencies:
-------------
- Cairo graphics library (pycairo) for high-quality rendering
- PIL (Pillow) for image format conversion and compatibility
- Standard library: random, math, dataclasses, typing

Usage:
------
Run this file directly to see a demonstration of the game in action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable, NamedTuple
import random
import math
import logging

from PIL import Image
import os

# ========================
# LOGGING CONFIGURATION
# ========================
# Set ENABLE_LOGGING = True to enable logging, False to disable
ENABLE_LOGGING = True

def setup_logging():
    """Configure logging for the environment module."""
    if ENABLE_LOGGING:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('set_env.log')
            ]
        )
    else:
        logging.disable(logging.CRITICAL)

setup_logging()
logger = logging.getLogger(__name__)

# Import Cairo board generator (required)
try:
    from .Utils.generate_board import generate_board_image
except ImportError:
    # Handle direct execution (python env.py)
    from Utils.generate_board import generate_board_image

# =====================
# Core rules & encoding
# =====================
FEATURE_DIM = 4
FEATURE_CARDINALITY = 3
DECK_SIZE = 81
INITIAL_BOARD_SIZE = 12
BOARD_INCREASE_STEP = 3
MAX_BOARD_SIZE = 21
REQUIRE_INITIAL_SET = True

# Feature index meanings for decoding (0..2 values for each)
F_NUMBER, F_COLOR, F_SHADING, F_SHAPE = 0, 1, 2, 3

# Color palette (RGB)
RGB_RED = (220, 60, 60)
RGB_GREEN = (40, 170, 80)
RGB_PURPLE = (120, 60, 190)
# Defaults; can be overridden via RenderCfg or palette sampling
CARD_BG_DEFAULT = (245, 245, 245)
BOARD_BG_DEFAULT = (235, 235, 240)
INDEX_BG = (0, 0, 0, 160)  # semi-transparent for index tag
INDEX_FG = (255, 255, 255, 255)

# --------------
# Encoders/decoders
# --------------

def code_to_vec(code: int) -> Tuple[int, int, int, int]:
    """Decode base-3 card code -> (number, color, shading, shape), each in {0,1,2}.
    Order: NUMBER, COLOR, SHADING, SHAPE.
    """
    assert 0 <= code < DECK_SIZE
    n = code % 3
    c = (code // 3) % 3
    s = (code // 9) % 3
    h = (code // 27) % 3
    return (n, c, s, h)


def vec_to_code(vec: Tuple[int, int, int, int]) -> int:
    n, c, s, h = vec
    assert all(v in (0, 1, 2) for v in vec)
    return n + 3 * c + 9 * s + 27 * h


# ---------
# Rule check
# ---------

def is_set(a: int, b: int, c: int) -> bool:
    """Classic SET predicate on card codes."""
    logger.debug(f"Checking if cards {a}, {b}, {c} form a valid set")
    if len({a, b, c}) != 3:
        logger.debug(f"Cards {a}, {b}, {c} are not distinct")
        return False
    va = code_to_vec(a)
    vb = code_to_vec(b)
    vc = code_to_vec(c)
    for i in range(FEATURE_DIM):
        # In ternary, sum % 3 == 0 iff all same or all different
        if (va[i] + vb[i] + vc[i]) % 3 != 0:
            logger.debug(f"Cards {a}, {b}, {c} fail set rule at feature {i}")
            return False
    logger.debug(f"Cards {a}, {b}, {c} form a valid set")
    return True


def all_sets(codes: Iterable[int]) -> List[Tuple[int, int, int]]:
    arr = list(codes)
    logger.debug(f"Finding all sets in {len(arr)} cards")
    out: List[Tuple[int, int, int]] = []
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if is_set(arr[i], arr[j], arr[k]):
                    out.append((arr[i], arr[j], arr[k]))
    logger.debug(f"Found {len(out)} valid sets")
    return out


@dataclass(frozen=True)
class RenderCfg:
    card_w: int = 220
    card_h: int = 320
    card_radius: int = 24
    pad: int = 24  # padding between cards
    grid_rows: int = 3  # 3 cards per column
    stripe_gap: int = 8
    index_tag_r: int = 14  # small index pill radius
    card_bg: Tuple[int, int, int] = CARD_BG_DEFAULT
    board_bg: Tuple[int, int, int] = BOARD_BG_DEFAULT

    def grid_cols(self, num_cards: int) -> int:
        """Calculate number of columns needed for given number of cards."""
        import math
        return math.ceil(num_cards / self.grid_rows)


def render_board(codes: List[int], cfg: RenderCfg, overlay_indices: bool = True) -> Image.Image:
    """Render board using Cairo graphics only."""
    logger.debug(f"Rendering board with {len(codes)} cards, overlay_indices={overlay_indices}")
    image = generate_board_image(codes, card_width=cfg.card_w, 
                                card_height=cfg.card_h, margin=cfg.pad, 
                                cols=cfg.grid_cols(len(codes)), rows=cfg.grid_rows,
                                overlay_indices=overlay_indices)
    logger.debug(f"Board rendered successfully, image size: {image.size}")
    return image


# ==============================
# Environment: state & operations
# ==============================

class EnvReturn(NamedTuple):
    message: Optional[str]
    image: Optional[Image.Image]


class SetEnv:
    """SET Zen environment with image-only board I/O.

    Indexing: selection expects 0-based positions into the current board. The
    rendered board image overlays human-friendly 1-based labels; callers should
    convert from a displayed label n to index n-1 before calling `select`.
    """

    def __init__(self, seed: Optional[int] = None, require_initial_set: bool = REQUIRE_INITIAL_SET):
        logger.info(f"Initializing SetEnv with seed={seed}, require_initial_set={require_initial_set}")
        self.cfg = RenderCfg()
        self.rng = random.Random(seed)
        self.require_initial_set = require_initial_set
        self._full_deck: List[int] = list(range(DECK_SIZE))
        self.deck: List[int] = []
        self.board: List[int] = []
        self.discard: List[int] = []
        logger.debug("SetEnv initialization complete, calling reset()")
        self.reset(seed)

    # --------
    # Lifecycle
    # --------
    def reset(self, seed: Optional[int] = None) -> Image.Image:
        logger.info(f"Resetting game with seed={seed}")
        if seed is not None:
            self.rng.seed(seed)
            logger.debug(f"RNG seeded with {seed}")
        # Fresh shuffled deck
        self.deck = list(range(DECK_SIZE))
        self.rng.shuffle(self.deck)
        logger.debug(f"Deck shuffled, first 5 cards: {self.deck[:5]}")
        # Deal initial board with guarantee
        attempts = 0
        while True:
            attempts += 1
            logger.debug(f"Dealing initial board, attempt {attempts}")
            self.board = self.deck[:INITIAL_BOARD_SIZE]
            self.deck = self.deck[INITIAL_BOARD_SIZE:]
            logger.debug(f"Initial board: {self.board}")
            if not self.require_initial_set:
                logger.debug("No initial set requirement, proceeding")
                break
            sets_here = all_sets(self.board)
            if len(sets_here) > 0:
                logger.info(f"Initial board contains {len(sets_here)} valid sets")
                break
            # resample: reshuffle and try again (safety capped)
            logger.debug(f"No valid sets found in initial board, attempt {attempts}")
            if attempts > 20:
                logger.warning(f"Reached maximum attempts ({attempts}), forcing a valid set")
                self._force_board_contains_set()
                break
            self.deck = list(range(DECK_SIZE))
            self.rng.shuffle(self.deck)
            logger.debug("Reshuffled deck for next attempt")
        self.discard = []
        logger.info(f"Game reset complete. Board size: {len(self.board)}, Deck size: {len(self.deck)}")
        return self.get_board_image()

    def _force_board_contains_set(self):
        # Replace the last card to complete a set with the first two
        logger.debug("Forcing board to contain a valid set")
        a, b = self.board[0], self.board[1]
        va, vb = code_to_vec(a), code_to_vec(b)
        need = tuple((6 - va[i] - vb[i]) % 3 for i in range(FEATURE_DIM))
        c = vec_to_code(need)
        logger.debug(f"Cards {a}, {b} need {c} to complete a set")
        if c in self.board:
            logger.debug(f"Required card {c} already on board")
            return
        if c in self.deck:
            self.deck.remove(c)
            logger.debug(f"Removed card {c} from deck")
        old_card = self.board[-1]
        self.board[-1] = c
        logger.debug(f"Replaced card {old_card} with {c} to force valid set")

    # -------
    # Imaging
    # -------
    def get_board_image(self) -> Image.Image:
        """Get the current board as an image using Cairo rendering."""
        logger.debug(f"Generating board image for {len(self.board)} cards")
        return render_board(self.board, self.cfg, overlay_indices=True)

    # -------
    # Actions
    # -------
    def select(self, indices: Tuple[int, int, int]) -> EnvReturn:
        """Agent proposes a triple of board indices (0-based).
        Contract:
          - NOT a Set: return (error text, current board image)
          - IS a Set: remove/replace, then return (success text, NEW board image)
        """
        logger.info(f"Select action called with indices: {indices}")
        if len(set(indices)) != 3:
            logger.debug(f"Indices {indices} are not distinct")
            return EnvReturn("Error: provide three distinct indices.", self.get_board_image())
        try:
            codes = [self.board[i] for i in indices]
            logger.debug(f"Selected cards: {codes} at indices {indices}")
        except IndexError as e:
            logger.debug(f"Index out of range error: {e}")
            return EnvReturn("Error: index out of range.", self.get_board_image())

        a, b, c = codes
        if not is_set(a, b, c):
            logger.info(f"Invalid set selected: cards {a}, {b}, {c}")
            return EnvReturn("Not a Set. Try again.", self.get_board_image())

        # Valid Set: remove from board (delete highest index first)
        logger.info(f"Valid set found: cards {a}, {b}, {c}")
        removed_cards = [self.board[i] for i in sorted(indices)]
        for i in sorted(indices, reverse=True):
            del self.board[i]
        logger.debug(f"Removed cards {removed_cards} from board")
        
        # Refill up to 12 if possible
        cards_added = 0
        while len(self.board) < INITIAL_BOARD_SIZE and self.deck:
            new_card = self.deck.pop(0)
            self.board.append(new_card)
            cards_added += 1
            logger.debug(f"Added card {new_card} to refill board")
        
        logger.info(f"Set removed successfully. Board size: {len(self.board)}, Cards added: {cards_added}")
        return EnvReturn("Success: valid Set removed.", self.get_board_image())

    def deal_three(self) -> EnvReturn:
        """Agent requests 3 more cards. Adds +3 up to MAX_BOARD_SIZE."""
        logger.info(f"Deal three requested. Current board size: {len(self.board)}")
        if len(self.board) >= MAX_BOARD_SIZE:
            logger.debug(f"Board at maximum size ({MAX_BOARD_SIZE}), cannot deal more")
            return EnvReturn("Error: board is at maximum size.", self.get_board_image())
        add = min(BOARD_INCREASE_STEP, MAX_BOARD_SIZE - len(self.board))
        logger.debug(f"Attempting to add {add} cards")
        if len(self.deck) < add:
            logger.debug(f"Deck has {len(self.deck)} cards, cannot deal {add} more")
            return EnvReturn("Error: deck depleted; cannot deal more cards.", self.get_board_image())
        new_cards = self.deck[:add]
        self.board.extend(new_cards)
        self.deck = self.deck[add:]
        logger.info(f"Successfully dealt {add} cards: {new_cards}. New board size: {len(self.board)}")
        return EnvReturn("Success: dealt 3 new cards.", self.get_board_image())


# =====
# Demo
# =====
if __name__ == "__main__":
    # Demo with Cairo rendering
    env = SetEnv(seed=42)
    img = env.get_board_image()
    img.save("board_initial.png")
    print("Saved initial board to board_initial.png")

    # Try a random triple (for demo only)
    move = (0, 1, 2)
    msg, img = env.select(move)
    print(msg)
    if img:
        img.save("board_after_move.png")

    # Deal three
    msg, img = env.deal_three()
    print(msg)
    if img:
        img.save("board_after_deal.png")
