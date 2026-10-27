"""When the gender model should answer "uncertain" instead of guessing.

Used by the API and the Streamlit app so both behave the same.
"""
from pathlib import Path

# Worldwide first names (sources in the file's header). Not in the list -> "uncertain".
_lines = (Path(__file__).parent / "first_names.txt").read_text(encoding="utf-8").splitlines()
KNOWN_NAMES = {n for n in _lines if n and not n.startswith("#")}

# Below this top probability -> "uncertain". Chosen on never-seen names, see reports/MODEL_EVALUATION.md.
THRESHOLD = 0.60


def name_error(name):
    """Return an error message for input that is not a name at all, else None."""
    if not name.replace(" ", "").replace("-", "").replace("'", "").isalpha():
        return "name must contain only letters, spaces, hyphens or apostrophes"
    if len(name.split()[0]) < 2:
        return "first name must be at least 2 letters"
    return None


def decide(first, probs):
    """first: lowercase first name, probs: {class: probability} -> (gender, note)."""
    best = max(probs, key=probs.get)
    if first not in KNOWN_NAMES:
        return "uncertain", f"'{first}' is not a recognised first name, so no gender is given."
    if probs[best] < THRESHOLD:
        return "uncertain", f"The model is not confident enough (below {THRESHOLD})."
    return best, "Predicted from first name only; users may choose not to disclose gender."
