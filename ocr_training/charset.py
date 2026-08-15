"""Character vocabulary and Indian registration-format normalisation for OCR.

The synthetic renderer never draws the letter ``I`` so that ``I``/``1`` can not be
confused during generation. Auditing the 2,122 reviewed real crops showed ``I``
appears in exactly two labels and both are annotation noise, so the model vocab
follows the renderer: 35 symbols, no ``I``.

``O`` is different. It is a legitimate state letter (``OD`` = Odisha) and a
legitimate series letter, but at a digit position it is always a misread ``0``.
:func:`normalise_registration` resolves that positionally instead of globally.
"""

from __future__ import annotations

import re
from itertools import combinations

#: Model vocabulary. Index 0 is reserved by the CTC loss for the blank symbol,
#: so a character's class index is ``VOCAB.index(ch) + 1``.
VOCAB = "0123456789ABCDEFGHJKLMNOPQRSTUVWXYZ"
BLANK_INDEX = 0
NUM_CLASSES = len(VOCAB) + 1

_CHAR_TO_INDEX = {ch: i + 1 for i, ch in enumerate(VOCAB)}
_INDEX_TO_CHAR = {i + 1: ch for i, ch in enumerate(VOCAB)}

#: Modern BS-standard registration: state, 2-digit district, series, 4 digits.
#: Repairs must land on this shape, because it is the only one strict enough to
#: make a corrected reading unambiguous.
CANONICAL_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{0,3}[0-9]{4}$")

#: Bharat-series plates run digits first.
BH_PATTERN = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")

#: Older and state-specific layouts. Accepted only when the annotator's reading
#: already parses, never as the target of a repair.
LEGACY_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{1,4}$")

#: Glyph pairs that annotators and OCR models routinely swap. Each entry maps a
#: character to the alternatives worth trying when the string does not parse.
_CONFUSIONS = {
    "O": "0",
    "0": "O",
    "I": "1",
    "1": "I",
    "Q": "0",
    "S": "5",
    "5": "S",
    "B": "8",
    "8": "B",
    "Z": "2",
    "2": "Z",
}


def encode(text: str) -> list[int]:
    """Map a registration string to CTC class indices."""
    return [_CHAR_TO_INDEX[ch] for ch in text]


def decode(indices) -> str:
    """Map class indices back to characters, ignoring the blank."""
    return "".join(_INDEX_TO_CHAR[i] for i in indices if i != BLANK_INDEX)


def ctc_greedy_decode(indices) -> str:
    """Collapse a raw CTC argmax path into a registration string."""
    out: list[str] = []
    previous = None
    for index in indices:
        if index != previous and index != BLANK_INDEX:
            out.append(_INDEX_TO_CHAR[index])
        previous = index
    return "".join(out)


def _strip(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def normalise_registration(raw: str) -> tuple[str | None, str]:
    """Return ``(cleaned_text, status)`` for one annotated registration.

    ``status`` is ``exact`` when the annotator's own reading already parses,
    ``repaired`` when exactly one confusable-glyph substitution turns it into a
    canonical registration, and ``rejected`` otherwise.

    Repair is deliberately conservative. These labels become the real-image test
    set, where a confidently wrong label is worse than a dropped one: it would
    be scored as a model error forever. So a repair is accepted only when the
    minimal edit distance yields a *single* canonical candidate. Ties are
    ambiguous by definition and get quarantined instead of guessed.
    """
    text = _strip(raw)
    if not text:
        return None, "rejected"

    # Trust an unmodified reading that already parses, including legacy layouts.
    if CANONICAL_PATTERN.match(text) or BH_PATTERN.match(text) or LEGACY_PATTERN.match(text):
        return _drop_i(text), "exact"

    positions = [i for i, ch in enumerate(text) if ch in _CONFUSIONS]
    if not positions or len(positions) > 12:
        return None, "rejected"

    for edits in range(1, len(positions) + 1):
        candidates = set()
        for combo in combinations(positions, edits):
            chars = list(text)
            for pos in combo:
                chars[pos] = _CONFUSIONS[text[pos]]
            candidate = "".join(chars)
            if CANONICAL_PATTERN.match(candidate) or BH_PATTERN.match(candidate):
                candidates.add(candidate)
        if len(candidates) == 1:
            return _drop_i(candidates.pop()), "repaired"
        if candidates:
            # More than one equally-plausible reading at this edit distance.
            return None, "ambiguous"
    return None, "rejected"


def _drop_i(text: str) -> str:
    """Fold ``I`` onto ``1`` so every label lives inside :data:`VOCAB`."""
    return text.replace("I", "1")


def in_vocab(text: str) -> bool:
    return bool(text) and all(ch in _CHAR_TO_INDEX for ch in text)
