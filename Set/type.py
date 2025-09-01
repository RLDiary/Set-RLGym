"""
type.py
========
Core type objects and lightweight value classes for a SET (card game) engine.

Design goals
------------
- Deterministic & training-friendly: explicit RNG seeds; stable, hashable values.
- Safety: frozen dataclasses / Enums; explicit error types; clear invariants.
- Minimal but complete: just the primitives—no game loop, no I/O.
- Zen mode only (single-player), but leave room for extensions.

Key invariants
--------------
- A 'Set' is any 3 cards where, for each attribute, the values are either all
  the same or all different.
- A fresh 12-card 'Board' produced by the engine must contain >= 1 valid Set
  when the deal policy requires it (default).
- Cards are unique (no duplicates) within a Deck, Board, and Discard.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import (
    Iterable, Iterator, Tuple, FrozenSet, Sequence, NewType, Literal, Optional,
    Protocol, Callable, NamedTuple, TypedDict, runtime_checkable
)
from uuid import UUID

class TypesError(Exception):
    """Base error for type and invariant violations."""


class InvalidCardError(TypesError):
    """Raised when a Card has invalid or inconsistent attributes/ID."""


class DuplicateCardError(TypesError):
    """Raised when duplicate cards appear where uniqueness is required."""


class InvalidSetError(TypesError):
    """Raised when a CardTriple fails the Set predicate but is used as a Set."""


class InvariantViolation(TypesError):
    """
    Raised when engine-level invariants fail (e.g., board must contain >=1 Set
    under current DealPolicy but doesn't).
    """

__all__ = [
    # IDs & aliases
    "Seed", "CardId", "BoardIndex",
    # Enums
    "Number", "Color", "Shape", "Shading",
    # Core values
    "Card", "CardTriple", "SetTriple",
    "Deck", "Board", "Discard",
    # Policies & settings
    "DealPolicy", "RngSpec", "EngineConfig", "ZenMode",
    # Protocols
    "RandomLike", "SetPredicate",
    # Errors
    "TypesError", "InvalidCardError", "DuplicateCardError", "InvalidSetError",
    "InvariantViolation",
    # Constants
    "STANDARD_DECK_SIZE", "BOARD_MIN", "BOARD_STANDARD",
    "ALL_NUMBERS", "ALL_COLORS", "ALL_SHAPES", "ALL_SHADINGS",
]

# ---------- IDs & Aliases ----------

Seed = NewType("Seed", int)                 # Training-friendly RNG seed
CardId = NewType("CardId", int)             # 0..80-1 for canonical 81-card deck
BoardIndex = NewType("BoardIndex", int)     # 0-based position on board grid

class Number(Enum):
    ONE = 1
    TWO = 2
    THREE = 3


class Color(Enum):
    RED = "red"
    GREEN = "green"
    PURPLE = "purple"


class Shape(Enum):
    DIAMOND = "diamond"
    SQUIGGLE = "squiggle"
    OVAL = "oval"


class Shading(Enum):
    SOLID = "solid"
    STRIPED = "striped"
    OPEN = "open"

ALL_NUMBERS: Tuple[Number, ...] = (Number.ONE, Number.TWO, Number.THREE)
ALL_COLORS: Tuple[Color, ...] = (Color.RED, Color.GREEN, Color.PURPLE)
ALL_SHAPES: Tuple[Shape, ...] = (Shape.DIAMOND, Shape.SQUIGGLE, Shape.OVAL)
ALL_SHADINGS: Tuple[Shading, ...] = (Shading.SOLID, Shading.STRIPED, Shading.OPEN)

STANDARD_DECK_SIZE: int = 81
BOARD_MIN: int = 12
BOARD_STANDARD: int = 12

@dataclass(frozen=True, slots=True)
class Card:
    """
    Immutable card value.

    Canonical mapping
    CardId encodes the 4 ternary attributes:
       id = n_idx + 3*c_idx + 9*s_idx + 27*h_idx  (each idx in {0,1,2})
    """
    number: Number
    color: Color
    shape: Shape
    shading: Shading
    id: Optional[CardId] = None  # Filled by deck constructor if not provided

    def as_tuple(self) -> Tuple[Number, Color, Shape, Shading]:
        return (self.number, self.color, self.shape, self.shading)

# A triple of cards (unordered for set-theory; use FrozenSet for hashing)
CardTriple = FrozenSet[Card]


@dataclass(frozen=True, slots=True)
class SetTriple:
    """
    Represents a validated 'Set' (exactly three distinct Cards).
    Construction SHOULD verify the set predicate.
    """
    cards: CardTriple  # len == 3, predicate == True

    def __post_init__(self):
        if len(self.cards) != 3:
            raise InvalidSetError("SetTriple must contain exactly 3 cards.")
        
@dataclass(frozen=True, slots=True)
class Deck:
    """
    Deck as an ordered, immutable sequence of unique Cards.
    The engine may treat this as a source for dealing; iteration yields in order.
    """
    cards: Tuple[Card, ...]   # length typically 81; unique by attributes/id

    def __iter__(self) -> Iterator[Card]:
        return iter(self.cards)

    def __len__(self) -> int:
        return len(self.cards)
    
@dataclass(frozen=True, slots=True)
class Board:
    """
    Immutable snapshot of the current board state.

    - 'cells' is a position-indexed tuple (for grid rendering and training labels).
    - Empty cells are represented by None (after sets are removed without refill).
    - 'present' contains the set of cards currently on the board (no None).
    - 'indices' contains valid occupied positions.
    """
    cells: Tuple[Optional[Card], ...]              # fixed capacity; e.g., 12, 15
    present: FrozenSet[Card]                       # derived: cards without None
    indices: FrozenSet[BoardIndex]                 # derived: occupied indices

    def size(self) -> int:
        return len(self.cells)

    def occupied(self) -> int:
        return len(self.present)

@dataclass(frozen=True, slots=True)
class Discard:
    """
    Immutable bag of removed cards. Ordering does not matter.
    """
    cards: FrozenSet[Card]

class DealPolicy(NamedTuple):
    """
    Dealing/board invariants.

    ensure_at_least_one_set:
        If True, the dealing algorithm MUST (re)sample until the visible board
        has >= 1 valid set (subject to max_attempts). For training, this avoids
        empty-start states.

    board_capacity:
        Number of visible cells (12 typical; allow 15 if your engine supports
        adding 3 when no set is visible).

    max_resample_attempts:
        Safety limit to prevent infinite loops in adversarial RNG settings.
    """
    ensure_at_least_one_set: bool
    board_capacity: int
    max_resample_attempts: int

class RngSpec(NamedTuple):
    """
    RNG specification for deterministic runs and dataset generation.
    - 'seed' is mandatory for reproducibility.
    - 'source' optionally records upstream UUID (e.g., dataset shard ID).
    """
    seed: Seed
    source: Optional[UUID] = None

@dataclass(frozen=True, slots=True)
class ZenMode:
    """
    Zen (single-player) metadata. Extendable for timers, scoring, curriculum.
    """
    # Example hooks for learning setups:
    target_min_sets: int = 0           # for curriculum tasks (optional)
    time_limit_s: Optional[float] = None

@dataclass(frozen=True, slots=True)
class EngineConfig:
    """
    Static configuration for a Zen game instance or dataset generator.
    """
    rng: RngSpec
    deal_policy: DealPolicy = DealPolicy(
        ensure_at_least_one_set=True,
        board_capacity=BOARD_STANDARD,
        max_resample_attempts=5,
    )
    zen: "ZenMode" = None  # filled post-init or by factory


class CardJSON(TypedDict):
    number: Literal[1, 2, 3]
    color: Literal["red", "green", "purple"]
    shape: Literal["diamond", "squiggle", "oval"]
    shading: Literal["solid", "striped", "open"]
    id: int  # 0..80


class BoardJSON(TypedDict):
    capacity: int
    cells: Sequence[Optional[int]]  # card IDs or None


class ConfigJSON(TypedDict):
    seed: int
    ensure_at_least_one_set: bool
    board_capacity: int
    max_resample_attempts: int

@runtime_checkable
class RandomLike(Protocol):
    """
    Minimal interface expected from RNG providers.
    Implemented by 'random.Random', 'numpy.random.Generator', etc.
    """
    def shuffle(self, x: list) -> None: ...
    def randint(self, a: int, b: int) -> int: ...  # inclusive


@runtime_checkable
class SetPredicate(Protocol):
    """
    Strategy for determining whether 3 cards form a Set.
    """
    def __call__(self, a: Card, b: Card, c: Card) -> bool: ...