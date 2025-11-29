# main.py – Arcana Herbarium Chatbot (cleaned + improved)
# ============================================================
# Ethnomedicinal Plant Chatbot (Mindanao, Philippines)
#
# Console-based chatbot that answers questions about
# ethnomedicinal plants found in Mindanao, Philippines.
#
# Core behavior:
#   1. Load plant data from a CSV via helper functions in vector.py.
#   2. Prefer direct CSV answers whenever possible (no AI guessing):
#        - Single-plant questions      → detailed plant summary.
#        - Multi-plant list questions  → grouped plants by disease/parts.
#        - Condition questions         → clear YES/NO per plant.
#        - Prep-method questions       → preparation, dosage, frequency.
#   3. Only when needed, call a local LLM (LLaMA 3.2 via Ollama)
#      limited strictly to the dataset text.
#   4. Display everything nicely using Rich panels in the terminal.
# ============================================================

from __future__ import annotations

# ------------------------------
# 1. IMPORTS
# ------------------------------
import re
from typing import List, Dict, Set, Tuple, Optional

# LLM + prompt
try:
    from langchain_ollama.llms import OllamaLLM           # Connects to local LLaMA via Ollama.
    from langchain_core.prompts import ChatPromptTemplate  # Helps build prompts with placeholders.
except Exception:  # makes this file importable even if libs are missing (useful for testing)
    OllamaLLM = None
    ChatPromptTemplate = None

# Dataset helpers / vector store
from vector import (
    retrieve_with_threshold,              # Vector search over CSV text chunks.
    get_known_plant_names,                # All plant names in the dataset (local + scientific).
    get_raw_records_for_name,             # All CSV rows for a given plant name.
    search_records_by_disease_keywords,   # CSV search by "Disease used on".
    search_records_by_parts_keywords,     # CSV search by "Parts used".
    get_all_disease_phrases,              # All distinct "Disease used on" phrases.
    get_dataset_vocabulary,               # All tokens appearing in the dataset.
    get_name_to_scientific_map,           # Alias → canonical scientific name mapping.
)

# Pretty console output
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

# Language detection
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

console = Console()

# ------------------------------
# 2. LANGUAGE DETECTION
# ------------------------------

# Simple Bisaya token heuristics (extend as needed)
BISAYA_TOKENS: Set[str] = {
    "unsa", "unsaon", "asa", "ni", "na", "kaon", "dili", "ngano",
    "kumusta", "salamat", "bughat", "gani",
}

def detect_language(text: str) -> str:
    """
    Return one of: 'english', 'tagalog', 'bisaya'.
    - Uses token heuristic for Bisaya, langdetect for Tagalog/English.
    """
    if not text or not text.strip():
        return "english"

    low = text.lower()

    # If we spot any clearly Bisaya word, call it Bisaya.
    if any(tok in low.split() for tok in BISAYA_TOKENS):
        return "bisaya"

    # Fallback to langdetect.
    try:
        lang = detect(text)
    except Exception:
        return "english"

    if lang == "tl":
        return "tagalog"
    if lang.startswith("en"):
        return "english"
    # Unknown language → treat as English for now
    return "english"


# ------------------------------
# 3. DATASET-DERIVED CONSTANTS
# ------------------------------

# All plant names present in the CSV (local + scientific, lowercased).
KNOWN_PLANT_NAMES: Set[str] = get_known_plant_names()

# All phrases from the "Disease used on" column.
ALL_DISEASE_PHRASES: List[str] = get_all_disease_phrases()

# All distinct tokens appearing anywhere in the CSV;
# used to detect questions that are "out of scope".
DATASET_VOCAB: Set[str] = get_dataset_vocabulary()

# Mapping from many aliases → canonical scientific name (exactly as in CSV).
NAME_TO_SCI: Dict[str, str] = get_name_to_scientific_map()


# ------------------------------
# 4. GENERAL TEXT HELPERS
# ------------------------------

def _clean_field(val) -> str:
    """Utility: treat NaN/None as empty and normalize whitespace."""
    s = str(val)
    if s.lower() == "nan":
        return ""
    return re.sub(r"\s+", " ", s).strip()


def highlight_keywords(text: str, keywords: List[str]) -> str:
    """
    Bold (Markdown **word**) any of the given keywords, case-insensitive.
    """
    if not keywords:
        return text

    escaped = [re.escape(kw) for kw in keywords if kw]
    if not escaped:
        return text

    pattern = r"\b(" + "|".join(escaped) + r")\b"

    def repl(match):
        return f"**{match.group(1)}**"

    return re.sub(pattern, repl, text, flags=re.IGNORECASE)


def tidy_response(text: str) -> str:
    """
    Lightly reformat model output so bullets look nice in the terminal.
    """
    text = text.replace(" -", "\n-")
    text = text.replace("•", "\n•")
    return text.strip()


def print_answer(text: str) -> None:
    """Show a successful answer in a green Rich panel."""
    panel = Panel(Markdown(text), title="🌿 ANSWER", border_style="green", padding=(1, 2))
    console.print(panel)


def print_error(text: str) -> None:
    """Show an error or warning in a red Rich panel."""
    panel = Panel(text, title="⚠️ ERROR", border_style="red", padding=(1, 2))
    console.print(panel)


# ------------------------------
# 5. PLANT NAME HELPERS
# ------------------------------

def find_plant_in_text(text: str) -> Optional[str]:
    """
    Scan free text and return the *scientific name* if we recognize
    any plant name / alias inside it.

    Uses NAME_TO_SCI which is built automatically from the dataset
    (scientific names + local names + normalized variants).
    """
    if not text:
        return None

    t = text.lower()
    best_match: Optional[str] = None
    best_len = 0

    for alias, sci_name in NAME_TO_SCI.items():
        if alias and alias in t and len(alias) > best_len:
            best_len = len(alias)
            best_match = sci_name

    return best_match

def find_all_plants_in_text(text: str) -> List[str]:
    """
    Return a list of canonical scientific names for ALL distinct plants
    mentioned in the user's text, in reading order.

    Uses the NAME_TO_SCI alias map (local names, hyphen/space variants, etc.).
    """
    if not text:
        return []

    t = text.lower()
    # (start_position, -alias_length, sci_name)
    matches: List[Tuple[int, int, str]] = []

    for alias, sci_name in NAME_TO_SCI.items():
        if not alias:
            continue
        pos = t.find(alias)
        if pos == -1:
            continue
        # earlier in the text first; for same position, longer alias first
        matches.append((pos, -len(alias), sci_name))

    if not matches:
        return []

    matches.sort(key=lambda x: (x[0], x[1]))

    seen_sci: Set[str] = set()
    ordered_scis: List[str] = []
    for _, _, sci_name in matches:
        if sci_name not in seen_sci:
            seen_sci.add(sci_name)
            ordered_scis.append(sci_name)

    return ordered_scis


def _get_plant_tokens(plant_name: str) -> Set[str]:
    """
    Collect tokens that belong to a plant's scientific name + all its
    known aliases. Used for detecting and trimming plant words from
    condition phrases.
    """
    tokens: Set[str] = set()

    # Tokens from the canonical scientific name
    for t in re.split(r"[^a-z]+", plant_name.lower()):
        if len(t) >= 3:
            tokens.add(t)

    # Tokens from aliases (local names, normalized forms)
    for alias, sci in NAME_TO_SCI.items():
        if sci != plant_name:
            continue
        for t in re.split(r"[^a-z]+", alias.lower()):
            if len(t) >= 3:
                tokens.add(t)

    return tokens


# ------------------------------
# 6. CONDITION / DISEASE HELPERS
# ------------------------------

# Stopwords used when extracting disease/condition phrases from questions.
DISEASE_STOPWORDS: Set[str] = {
    # generic question words
    "what", "which", "who", "where", "when", "why", "how",
    # generic verbs + helpers
    "is", "are", "was", "were", "be", "being", "been",
    "do", "does", "did",
    "can", "could", "would", "should", "may", "might",
    # articles / prepositions
    "in", "on", "at", "for", "from", "to", "of", "by", "with",
    "the", "a", "an", "this", "that", "these", "those",
    # plant words
    "plant", "plants", "tree", "trees", "herb", "herbs",
    # generic medical verbs we don't want to treat as diseases
    "treat", "treats", "treating", "treatment",
    "cure", "cures", "curing",
    "heal", "heals", "healing",
    "relief", "relieve", "relieves",
    "used", "use", "uses", "using",
    # generic question / command words
    "give", "show", "list", "provide", "fetch",
    "please", "kindly",
    # very generic symptom words
    "pain", "pains", "ache", "aches",
    # high-frequency non-condition words
    "all", "any", "some", "as",
    "medicinal", "part", "parts",
    "disease", "diseases",
    "condition", "conditions",
    "problem", "problems",
}


def clean_condition_label(label: str) -> str:
    """
    Normalize condition labels like 'and for anemia' → 'anemia'.
    """
    if not label:
        return ""

    s = label.strip()
    s = re.sub(r"^(and|or)\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(for\s+)+", "", s, flags=re.IGNORECASE)
    return s.strip()


def parse_is_used_for_question(query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse English questions like:
      'Is Andrographis paniculata used for cough?'
      'Can white flower treat jaundice?'

    into (plant_fragment, condition_fragment).
    """
    q = query.lower().replace("’", "'").strip()

    m = re.search(
        r"\b(?:is|are|can|could|may|might|will|would|does|do)\s+(.+?)\s+"
        r"(?:used for|used to treat|used as|good for|for treating|"
        r"to treat|treat(?:ing)?|treats?|helps? with|help with)\s+"
        r"([a-z\s\-]+)\??",
        q,
    )
    if not m:
        return None, None

    plant_fragment = m.group(1).strip(" ?.!,")
    cond_fragment = m.group(2).strip(" ?.!,")
    if not plant_fragment or not cond_fragment:
        return None, None

    return plant_fragment, cond_fragment


def extract_condition_phrases_from_query(query: str) -> List[str]:
    """
    Try to pull out the *condition phrase(s)* from the user's question.

    1) Prefer things after 'to treat X', 'treat X', 'helps with X'.
    2) Otherwise fall back to 'used for X', 'for X', 'as X'.
    3) If that still fails, return keyword tokens.
    """
    q = query.lower()
    q = q.replace("’", "'")

    # 1) Multi-word matching for some common phrases
    MULTIWORD_CONDITIONS = [
        "cancer therapeutics",
        "gas pain",
        "stomach pain",
        "high blood pressure",
        "high blood",
    ]
    found_multi: List[str] = []
    for phrase in MULTIWORD_CONDITIONS:
        if phrase in q:
            found_multi.append(phrase)
    if found_multi:
        return found_multi

    # 2a) Prefer patterns that explicitly mention treatment
    m = re.search(
        r"\b(?:to treat|treat(?:ing)?|treats?|helps? with|help with)\s+([a-z\s\-\"']+)",
        q,
    )

    # 2b) Otherwise, patterns with "used for / for / as"
    if not m:
        m = re.search(
            r"\b(?:used for|used as|good for|for treating|for|as)\s+([a-z\s\-\"']+)",
            q,
        )

    if m:
        cond = m.group(1)
        # Strip quotes/punctuation
        cond = re.sub(r"[^a-z\s\-]", " ", cond)
        cond = re.sub(r"\s+", " ", cond).strip()
        if cond:
            return [cond]

    # 3) Fallback: keyword tokens (last resort)
    cleaned = q
    for ch in "?,.!;:":
        cleaned = cleaned.replace(ch, " ")
    tokens = [t for t in cleaned.split() if t and t not in DISEASE_STOPWORDS]
    return tokens


def strip_plant_name_from_condition(plant_name: str, condition: str) -> str:
    """
    Example:
      plant_name = 'Peperomia pellucida (L.) Kunth'
      condition  = 'peperomia pellucida for skin burns'
      -> 'skin burns'
    """
    if not condition:
        return ""

    plant_tokens = _get_plant_tokens(plant_name)
    if not plant_tokens:
        return condition.strip()

    tokens = [t for t in re.split(r"[^a-z]+", condition.lower()) if t]
    if not tokens:
        return condition.strip()

    # Skip leading tokens that are part of the plant name or filler words.
    i = 0
    while i < len(tokens) and (
        tokens[i] in plant_tokens or tokens[i] in {"for", "the", "this", "that", "of"}
    ):
        i += 1

    if i == 0 or i >= len(tokens):
        # Nothing to trim or everything got trimmed → keep original.
        return condition.strip()

    return " ".join(tokens[i:]).strip()


def is_condition_actually_disease(plant_name: str, condition_phrases: List[str]) -> bool:
    """
    Decide if the extracted 'condition' phrases are *real conditions*
    or just another way of saying the plant name.

    - Return **False** if the condition tokens are basically only the
      plant name / its known synonyms (e.g. "lagundi").
    - Return **True** if there are tokens that are not part of the plant
      name (e.g. "cough", "bughat in women").
    """
    if not plant_name or not condition_phrases:
        return False

    plant_tokens = _get_plant_tokens(plant_name)

    if not plant_tokens:
        # If we somehow have no tokens for the plant, just assume
        # the condition really is a condition.
        return True

    # Collect tokens from the condition phrases
    condition_tokens: Set[str] = set()
    for cond in condition_phrases:
        for t in re.split(r"[^a-z]+", str(cond).lower()):
            if len(t) >= 3:
                condition_tokens.add(t)

    if not condition_tokens:
        return False

    # If *all* condition tokens are words from the plant name / aliases,
    # then it's not a real disease.
    only_name_tokens = condition_tokens.issubset(plant_tokens)
    return not only_name_tokens


def is_condition_question(query: str) -> bool:
    """
    True for yes/no style condition questions like (in English):
      'Can X treat anemia?'
      'Is X used for cough?'

    NOT for 'what is X used for?'
    """
    q = query.strip().lower()

    if q.startswith("what "):
        return False

    if not re.match(r"^(can|does|do|is|are|will|would|may|might|could)\b", q):
        return False

    if re.search(
        r"(treat|treats|treating|used for|used to treat|used as|good for|"
        r"helps? with|help with|relieve|relieves)",
        q,
    ):
        return True

    return False


# ------------------------------
# 7. PARTS-BASED HELPERS
# ------------------------------

PART_TERMS: List[str] = [
    "leaf", "leaves", "lf",
    "root", "roots",
    "bark",
    "stem", "stems",
    "whole plant", "whole-plant",
    "flower", "flowers",
    "fruit", "fruits",
    "seed", "seeds",
    "rhizome", "bulb",
]

# Map raw tokens -> (Full name, Short code)
PART_DISPLAY_MAP: Dict[str, Tuple[str, str]] = {
    # Bark
    "bk": ("Bark", "Bk"),
    "bark": ("Bark", "Bk"),

    # Leaf
    "lf": ("Leaf", "Lf"),
    "leaf": ("Leaf", "Lf"),
    "leaves": ("Leaf", "Lf"),

    # Root
    "rt": ("Root", "Rt"),
    "root": ("Root", "Rt"),

    # Whole plant
    "wp": ("Whole plant", "Wp"),
    "whole plant": ("Whole plant", "Wp"),

    # Flower
    "fl": ("Flower", "Fl"),
    "flower": ("Flower", "Fl"),

    # Fruit
    "fr": ("Fruit", "Fr"),
    "fruit": ("Fruit", "Fr"),

    # Seed
    "sd": ("Seed", "Sd"),
    "seed": ("Seed", "Sd"),

    # Rhizome / tuber etc.
    "rh": ("Rhizome", "Rh"),
    "rhizome": ("Rhizome", "Rh"),
    "tb": ("Tuber", "Tb"),
    "tuber": ("Tuber", "Tb"),
}


def standardize_parts_values(raw_values: List[str]) -> str:
    """
    Take a list of raw 'Parts used' values from the CSV and return
    a clean standardized string like:

        'Bark (Bk); Leaf (Lf)'

    It understands codes like 'Bk; Lf', words like 'leaf', or
    already-formatted 'Bark (Bk)' and merges everything nicely.
    """
    items: List[str] = []

    for val in raw_values:
        if not val:
            continue

        text = str(val)

        # split on common separators ; , /
        pieces = re.split(r"[;,/]+", text)
        for piece in pieces:
            p = piece.strip()
            if not p:
                continue

            # If it's already like "Bark (Bk)" keep that info
            m = re.match(r"([A-Za-z ]+)\(([A-Za-z]+)\)", p)
            if m:
                full = m.group(1).strip().title()
                code = m.group(2).strip()
            else:
                key = p.lower()
                if key in PART_DISPLAY_MAP:
                    full, code = PART_DISPLAY_MAP[key]
                else:
                    # Fallback: just prettify the text, no code
                    full, code = p.title(), ""

            label = f"{full} ({code})" if code else full
            items.append(label)

    # Deduplicate while preserving order
    seen = set()
    ordered: List[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            ordered.append(it)

    return "; ".join(ordered)


def is_parts_based_request(query: str) -> bool:
    """
    Return True if the question is clearly about plant parts.
    """
    q = query.lower()
    if "part " in q or "parts " in q or "parts used" in q:
        return True
    return any(term in q for term in PART_TERMS)


def extract_parts_keywords_from_query(query: str) -> List[str]:
    """
    From a question like:
      'List plants that use leaves as medicinal parts'
    return keywords that match the CSV "Parts used" column.
    """
    q = query.lower()
    found: Set[str] = set()

    PART_SYNONYMS: Dict[str, List[str]] = {
        "leaf": ["leaf", "leaves", "lf"],
        "leaves": ["leaf", "leaves", "lf"],
        "lf": ["leaf", "leaves", "lf"],
        "root": ["root", "roots"],
        "roots": ["root", "roots"],
        "whole plant": ["whole plant"],
        "bark": ["bark"],
        "stem": ["stem", "stems"],
        "flower": ["flower", "flowers"],
        "flowers": ["flower", "flowers"],
        "fruit": ["fruit", "fruits"],
        "fruits": ["fruit", "fruits"],
        "seed": ["seed", "seeds"],
        "seeds": ["seed", "seeds"],
        "rhizome": ["rhizome"],
        "bulb": ["bulb"],
    }

    for phrase, mapped in PART_SYNONYMS.items():
        if phrase in q:
            for m in mapped:
                found.add(m)

    if not found and "part" in q:
        found.update(["leaf", "leaves", "lf"])

    return sorted(found)


# ------------------------------
# 8. CSV-BASED ANSWER BUILDERS
# ------------------------------

FALLBACK_MSG = "I'm sorry, but I can't find information about that in the dataset."


def build_per_record_details(records: List[Dict]) -> str:
    """
    Build a detailed, row-by-row view that keeps the relationship between:
    Disease ↔ Parts used ↔ Preparation ↔ Dosage ↔ Frequency ↔ Side effects ↔ Source.
    """
    if not records:
        return ""

    lines: List[str] = []
    lines.append("**Dataset records grouped by indication:**")

    for idx, row in enumerate(records, start=1):
        disease = _clean_field(row.get("Disease used on", ""))
        parts_raw = _clean_field(row.get("Parts used", ""))
        prep = _clean_field(row.get("Preparation and Administration", ""))
        dosage = _clean_field(row.get("Quality of Dosage", ""))
        freq = _clean_field(row.get("Administration Frequency", ""))
        sides = _clean_field(row.get("Experienced adverse or side effects", ""))
        lit = _clean_field(row.get("Literature taken from", ""))
        page = _clean_field(
            row.get("Page where the data can be found on the literature", "")
        )

        # --- Header for this use case ---
        lines.append("")  # blank line
        header = f"**Indication {idx}**"
        if disease:
            header += f": {disease}"
        lines.append(header)

        # --- Linked details for this specific row ---
        if parts_raw:
            parts_std = standardize_parts_values([parts_raw])
            lines.append(f"- Parts used: {parts_std}")
        if prep:
            lines.append(f"- Preparation and Administration: {prep}")
        if dosage:
            lines.append(f"- Quality of Dosage: {dosage}")
        if freq:
            lines.append(f"- Administration Frequency: {freq}")
        if sides:
            lines.append(f"- Experienced adverse or side effects: {sides}")

        # Source for this exact row
        if lit or page:
            if page:
                page = re.sub(r"(?i)^page[: ]*", "", page).strip()
            src_bits = []
            if lit:
                src_bits.append(lit)
            if page:
                src_bits.append(f"page {page}")
            lines.append(f"- Source: {', '.join(src_bits)}")

    return "\n".join(lines)


def build_answer_from_records(records: List[Dict]) -> str:
    """
    Plain CSV summary for a plant (or a set of rows for that plant),
    followed by per-record details that preserve the relationships.
    """
    if not records:
        return FALLBACK_MSG

    def collect_unique(key: str) -> List[str]:
        seen, values = set(), []
        for row in records:
            v = _clean_field(row.get(key, ""))
            if v and v not in seen:
                seen.add(v)
                values.append(v)
        return values

    locals_ = collect_unique("Local Name")
    scis = collect_unique("Scientific Name")
    families = collect_unique("Family")
    parts_list = collect_unique("Parts used")
    diseases = collect_unique("Disease used on")
    preps = collect_unique("Preparation and Administration")
    dosages = collect_unique("Quality of Dosage")
    freqs = collect_unique("Administration Frequency")
    sides = collect_unique("Experienced adverse or side effects")

    local_main = locals_[0] if locals_ else ""
    sci_main = scis[0] if scis else ""

    def join_many(values: List[str]) -> str:
        return values[0] if len(values) == 1 else "; ".join(values)

    title = local_main or "Unknown local name"
    if sci_main:
        title = f"{title} ({sci_main})"

    summary_lines: List[str] = [f"**{title}**"]

    if families:
        summary_lines.append(f"- Family: {join_many(families)}")
    if parts_list:
        parts_text = standardize_parts_values(parts_list)
        summary_lines.append(f"- Parts used (overall): {parts_text}")
    if diseases:
        summary_lines.append(f"- Uses (Disease used on, overall): {join_many(diseases)}")
    if preps:
        summary_lines.append(
            f"- Preparation and administration (all records): {join_many(preps)}"
        )
    if dosages:
        summary_lines.append(
            f"- Quality of dosage (all records): {join_many(dosages)}"
        )
    if freqs:
        summary_lines.append(
            f"- Administration frequency (all records): {join_many(freqs)}"
        )
    if sides:
        summary_lines.append(
            f"- Reported side effects (any record): {join_many(sides)}"
        )

    # Aggregated sources (all rows)
    sources = set()
    for row in records:
        lit = _clean_field(row.get("Literature taken from", ""))
        page = _clean_field(
            row.get("Page where the data can be found on the literature", "")
        )
        if page:
            page = re.sub(r"(?i)^page[: ]*", "", page).strip()
        if lit or page:
            sources.add((lit, page))

    for idx, (lit, page) in enumerate(sorted(sources), start=1):
        if lit and page:
            summary_lines.append(f"- Source {idx}: {lit}, page {page}")
        elif lit:
            summary_lines.append(f"- Source {idx}: {lit}")
        elif page:
            summary_lines.append(f"- Source {idx}: page {page}")

    summary_block = "\n".join(summary_lines)
    detail_block = build_per_record_details(records)

    if not detail_block.strip():
        return summary_block
    return summary_block + "\n\n" + detail_block


def build_intro_answer_from_records(records: List[Dict]) -> str:
    """
    For "What is X?" or "Ano ang gamit ng X?" style questions:
      - Short descriptive paragraph.
      - High-level summary bullets.
      - Row-by-row details that show how each use is prepared/administered.
    """
    if not records:
        return FALLBACK_MSG

    def collect_unique(key: str) -> List[str]:
        seen, values = set(), []
        for row in records:
            v = _clean_field(row.get(key, ""))
            if v and v not in seen:
                seen.add(v)
                values.append(v)
        return values

    locals_ = collect_unique("Local Name")
    scis = collect_unique("Scientific Name")
    families = collect_unique("Family")
    diseases = collect_unique("Disease used on")
    parts = collect_unique("Parts used")
    preps = collect_unique("Preparation and Administration")
    dosages = collect_unique("Quality of Dosage")
    freqs = collect_unique("Administration Frequency")
    sides = collect_unique("Experienced adverse or side effects")

    local_main = locals_[0] if locals_ else ""
    sci_main = scis[0] if scis else ""

    if local_main and sci_main:
        plant_label = f"{local_main} ({sci_main})"
    elif sci_main:
        plant_label = sci_main
    elif local_main:
        plant_label = local_main
    else:
        plant_label = "This plant"

    # ---------- Intro paragraph ----------
    sentence_parts: List[str] = []
    sentence_parts.append(
        f"{plant_label} is an ethnomedicinal plant recorded in this Mindanao ethnomedicinal plant dataset."
    )

    if families:
        fam_text = families[0] if len(families) == 1 else "; ".join(families)
        sentence_parts.append(f"It belongs to the {fam_text} family.")

    if diseases:
        uses_text = diseases[0] if len(diseases) == 1 else "; ".join(diseases)
        sentence_parts.append(f"It is traditionally used for {uses_text}.")

    if parts:
        parts_text = standardize_parts_values(parts)
        sentence_parts.append(
            f"The recorded medicinal parts used include {parts_text}."
        )

    intro_paragraph = " ".join(sentence_parts)

    # ---------- High-level bullet block ----------
    lines: List[str] = ["**Detailed information from the dataset:**"]
    if local_main:
        lines.append(f"- Local Name: {local_main}")
    if sci_main:
        lines.append(f"- Scientific Name: {sci_main}")
    if families:
        lines.append(f"- Family: {'; '.join(families)}")
    if parts:
        parts_text = standardize_parts_values(parts)
        lines.append(f"- Parts used (overall): {parts_text}")
    if diseases:
        lines.append(f"- Uses (Disease used on, overall): {'; '.join(diseases)}")
    if preps:
        lines.append(f"- Preparation and Administration (overall): {'; '.join(preps)}")
    if dosages:
        lines.append(f"- Quality of Dosage (overall): {'; '.join(dosages)}")
    if freqs:
        lines.append(f"- Administration Frequency (overall): {'; '.join(freqs)}")
    if sides:
        lines.append(
            f"- Experienced adverse or side effects (overall): {'; '.join(sides)}"
        )

    # Aggregated sources
    sources = set()
    for row in records:
        lit = _clean_field(row.get("Literature taken from", ""))
        page = _clean_field(
            row.get("Page where the data can be found on the literature", "")
        )
        if page:
            page = re.sub(r"(?i)^page[: ]*", "", page).strip()
        if lit or page:
            sources.add((lit, page))

    for idx, (lit, page) in enumerate(sorted(sources), start=1):
        if lit and page:
            lines.append(f"- Source {idx}: {lit}, page {page}")
        elif lit:
            lines.append(f"- Source {idx}: {lit}")
        elif page:
            lines.append(f"- Source {idx}: page {page}")

    summary_block = "\n".join(lines)

    # ---------- Attach per-record details ----------
    detail_block = build_per_record_details(records)

    if detail_block.strip():
        return intro_paragraph + "\n\n" + summary_block + "\n\n" + detail_block
    else:
        return intro_paragraph + "\n\n" + summary_block


def build_condition_answer(plant_name: str, condition_phrases: List[str]) -> str:
    """
    YES/NO condition-specific answer for a single plant.
    """
    cleaned_conditions: List[str] = []
    for raw in condition_phrases or []:
        s = clean_condition_label(str(raw))
        if s:
            cleaned_conditions.append(s)

    if not cleaned_conditions:
        return FALLBACK_MSG

    main_condition = cleaned_conditions[0].strip()
    if not main_condition:
        return FALLBACK_MSG

    records = get_raw_records_for_name(plant_name)
    if not records:
        return FALLBACK_MSG

    def row_mentions_condition(row: Dict, cond: str) -> bool:
        disease_text = str(row.get("Disease used on", "")).lower()
        cond = cond.lower().strip()
        if not cond:
            return False
        if cond in disease_text:
            return True
        tokens = [
            t for t in re.split(r"[^a-z]+", cond)
            if t and t not in DISEASE_STOPWORDS
        ]
        return any(tok in disease_text for tok in tokens)

    filtered: List[Dict] = []
    for row in records:
        if any(row_mentions_condition(row, cond) for cond in cleaned_conditions):
            filtered.append(row)

    ref = records[0]
    local_main = _clean_field(ref.get("Local Name", ""))
    sci_main = _clean_field(ref.get("Scientific Name", ""))

    if local_main and sci_main:
        plant_label = f"{local_main} ({sci_main})"
    elif sci_main:
        plant_label = sci_main
    elif local_main:
        plant_label = local_main
    else:
        plant_label = plant_name

    # Case 1: plant exists but condition not mentioned.
    if not filtered:
        first_line = (
            f"The dataset does not mention **{main_condition}** "
            f"as an indication for {plant_label}."
        )
        general_info = build_answer_from_records(records)

        highlight_terms = set()
        for cond in cleaned_conditions:
            highlight_terms.add(cond)
            for piece in re.split(r"[;,/ ]+", cond):
                piece = piece.strip()
                if (
                    piece
                    and piece.lower() not in {"and", "or", "for"}
                    and len(piece) >= 4
                ):
                    highlight_terms.add(piece)

        combined = first_line + "\n\n" + general_info
        combined = highlight_keywords(combined, list(highlight_terms))
        return combined

    # Case 2: YES, plant is used for that condition.
    def collect_unique(rows: List[Dict], key: str) -> List[str]:
        seen, values = set(), []
        for r in rows:
            v = _clean_field(r.get(key, ""))
            if v and v not in seen:
                seen.add(v)
                values.append(v)
        return values

    diseases = collect_unique(filtered, "Disease used on")
    parts_list = collect_unique(filtered, "Parts used")
    preps = collect_unique(filtered, "Preparation and Administration")
    dosages = collect_unique(filtered, "Quality of Dosage")
    freqs = collect_unique(filtered, "Administration Frequency")
    sides = collect_unique(filtered, "Experienced adverse or side effects")

    def join_many(values: List[str]) -> str:
        return values[0] if len(values) == 1 else "; ".join(values)

    lines: List[str] = []
    lines.append(
        f"Yes. According to the dataset, {plant_label} can be used for **{main_condition}**."
    )
    lines.append("")

    if diseases:
        lines.append(f"- Disease used on: {join_many(diseases)}")
    if parts_list:
        parts_text = standardize_parts_values(parts_list)
        lines.append(
            f"- Parts used (for **{main_condition}**): {parts_text}"
        )
    if preps:
        lines.append(
            f"- Preparation and Administration (for **{main_condition}**): "
            f"{join_many(preps)}"
        )
    if dosages:
        lines.append(
            f"- Quality of Dosage (for **{main_condition}**): {join_many(dosages)}"
        )
    if freqs:
        lines.append(
            f"- Administration Frequency (for **{main_condition}**): "
            f"{join_many(freqs)}"
        )
    if sides:
        lines.append(f"- Reported side effects: {join_many(sides)}")

    # Sources for rows matching the condition.
    sources = set()
    for row in filtered:
        lit = _clean_field(row.get("Literature taken from", ""))
        page = _clean_field(
            row.get("Page where the data can be found on the literature", "")
        )
        if page:
            page = re.sub(r"(?i)^page[: ]*", "", page).strip()
        if lit or page:
            sources.add((lit, page))

    for idx, (lit, page) in enumerate(sorted(sources), start=1):
        if lit and page:
            lines.append(f"- Source {idx}: {lit}, page {page}")
        elif lit:
            lines.append(f"- Source {idx}: {lit}")
        elif page:
            lines.append(f"- Source {idx}: page {page}")

    # Highlight condition words.
    highlight_terms = set()
    for cond in cleaned_conditions:
        highlight_terms.add(cond)
        for piece in re.split(r"[;,/ ]+", cond):
            piece = piece.strip()
            if (
                piece
                and piece.lower() not in {"and", "or", "for"}
                and len(piece) >= 4
            ):
                highlight_terms.add(piece)

    final_text = "\n".join(lines)
    final_text = highlight_keywords(final_text, list(highlight_terms))
    return final_text


def build_prep_method_answer(plant_name: str, condition_phrases: List[str]) -> str:
    """
    Answer questions like (English):
      - "prep method for lagundi"
      - "what is the preparation method for lagundi to treat bughat in women?"

    Rules:
      - If no clear condition is given → return a *general* preparation summary
        for that plant (parts used, preparation, dosage, frequency).
      - If a condition is given:
          * If the dataset has rows for that plant + condition → show a short,
            condition-focused prep summary.
          * If the condition is NOT mentioned for that plant → say that clearly,
            then show the general prep info anyway.

    All answers are built **directly from the CSV**, no LLM.
    """
    records = get_raw_records_for_name(plant_name)
    if not records:
        return FALLBACK_MSG

    def _clean(val):
        s = str(val)
        if s.lower() == "nan":
            return ""
        return re.sub(r"\s+", " ", s).strip()

    def _collect_unique(rows: List[Dict], key: str) -> List[str]:
        seen = set()
        out: List[str] = []
        for r in rows:
            v = _clean(r.get(key, ""))
            if v and v not in seen:
                seen.add(v)
                out.append(v)
        return out

    ref = records[0]
    local_main = _clean(ref.get("Local Name", ""))
    sci_main = _clean(ref.get("Scientific Name", ""))

    if local_main and sci_main:
        plant_label = f"{local_main} ({sci_main})"
    elif sci_main:
        plant_label = sci_main
    elif local_main:
        plant_label = local_main
    else:
        plant_label = plant_name

    def _build_general_prep_block() -> str:
        parts   = _collect_unique(records, "Parts used")
        preps   = _collect_unique(records, "Preparation and Administration")
        dosages = _collect_unique(records, "Quality of Dosage")
        freqs   = _collect_unique(records, "Administration Frequency")

        lines: List[str] = []
        lines.append(
            f"Here is the general preparation information recorded for {plant_label}:"
        )

        if parts:
            parts_text = standardize_parts_values(parts)
            lines.append(f"- Parts used: {parts_text}")
        if preps:
            lines.append(f"- Preparation and Administration: {'; '.join(preps)}")
        if dosages:
            lines.append(f"- Quality of Dosage: {'; '.join(dosages)}")
        if freqs:
            lines.append(f"- Administration Frequency: {'; '.join(freqs)}")

        # Sources (literature + page) for all rows of this plant
        sources = set()
        for row in records:
            lit = _clean(row.get("Literature taken from", ""))
            page = _clean(
                row.get("Page where the data can be found on the literature", "")
            )
            if page:
                page = re.sub(r"(?i)^page[: ]*", "", page).strip()
            if lit or page:
                sources.add((lit, page))

        for idx, (lit, page) in enumerate(sorted(sources), start=1):
            if lit and page:
                lines.append(f"- Source {idx}: {lit}, page {page}")
            elif lit:
                lines.append(f"- Source {idx}: {lit}")
            elif page:
                lines.append(f"- Source {idx}: page {page}")

        return "\n".join(lines)

    # If user did NOT specify any condition → just return general prep info
    if not condition_phrases:
        return _build_general_prep_block()

    # Condition-aware branch.
    cleaned_conditions: List[str] = []
    for raw in condition_phrases:
        s = clean_condition_label(str(raw))
        if s:
            cleaned_conditions.append(s)

    trimmed_conditions: List[str] = []
    for cond in cleaned_conditions:
        trimmed = strip_plant_name_from_condition(plant_name, cond)
        if trimmed:
            trimmed_conditions.append(trimmed)
    cleaned_conditions = trimmed_conditions

    if not cleaned_conditions:
        return _build_general_prep_block()

    main_condition = cleaned_conditions[0]

    def _row_mentions_condition(row: Dict, cond: str) -> bool:
        disease_text = str(row.get("Disease used on", "")).lower()
        cond = cond.lower().strip()
        if not cond:
            return False

        if cond in disease_text:
            return True

        tokens = [
            t for t in re.split(r"[^a-z]+", cond)
            if t and t not in DISEASE_STOPWORDS
        ]
        return any(tok in disease_text for tok in tokens)

    condition_rows: List[Dict] = []
    for row in records:
        for cond in cleaned_conditions:
            if _row_mentions_condition(row, cond):
                condition_rows.append(row)
                break

    if not condition_rows:
        first_line = (
            f"The dataset does not mention **{main_condition}** "
            f"as an indication for {plant_label}."
        )
        general_block = _build_general_prep_block()

        highlight_terms = set()
        for cond in cleaned_conditions:
            highlight_terms.add(cond)
            for piece in re.split(r"[;,/ ]+", cond):
                piece = piece.strip()
                if (
                    piece
                    and piece.lower() not in {"and", "or", "for"}
                    and len(piece) >= 4
                ):
                    highlight_terms.add(piece)

        combined = first_line + "\n\n" + general_block
        combined = highlight_keywords(combined, list(highlight_terms))
        return combined

    # Condition IS mentioned
    parts   = _collect_unique(condition_rows, "Parts used")
    preps   = _collect_unique(condition_rows, "Preparation and Administration")
    dosages = _collect_unique(condition_rows, "Quality of Dosage")
    freqs   = _collect_unique(condition_rows, "Administration Frequency")

    lines: List[str] = []
    lines.append(
        f"Preparation information for {plant_label} "
        f"when used for **{main_condition}** (according to the dataset):"
    )

    if parts:
        parts_text = standardize_parts_values(parts)
        lines.append(
            f"- Parts used (for **{main_condition}**): {parts_text}"
        )
    if preps:
        lines.append(
            f"- Preparation and Administration (for **{main_condition}**): "
            f"{'; '.join(preps)}"
        )
    if dosages:
        lines.append(
            f"- Quality of Dosage (for **{main_condition}**): {'; '.join(dosages)}"
        )
    if freqs:
        lines.append(
            f"- Administration Frequency (for **{main_condition}**): {'; '.join(freqs)}"
        )

    # Sources for rows matching the condition
    sources = set()
    for row in condition_rows:
        lit = _clean(row.get("Literature taken from", ""))
        page = _clean(
            row.get("Page where the data can be found on the literature", "")
        )
        if page:
            page = re.sub(r"(?i)^page[: ]*", "", page).strip()
        if lit or page:
            sources.add((lit, page))

    for idx, (lit, page) in enumerate(sorted(sources), start=1):
        if lit and page:
            lines.append(f"- Source {idx}: {lit}, page {page}")
        elif lit:
            lines.append(f"- Source {idx}: {lit}")
        elif page:
            lines.append(f"- Source {idx}: page {page}")

    highlight_terms = set()
    for cond in cleaned_conditions:
        highlight_terms.add(cond)
        for piece in re.split(r"[;,/ ]+", cond):
            piece = piece.strip()
            if (
                piece
                and piece.lower() not in {"and", "or", "for"}
                and len(piece) >= 4
            ):
                highlight_terms.add(piece)

    final_text = "\n".join(lines)
    final_text = highlight_keywords(final_text, list(highlight_terms))
    return final_text


def build_multi_plant_grouped_answer(
    records_by_plant: Dict[str, List[Dict]],
    header: str,
) -> str:
    """
    Nicely format multi-plant answers (for disease- or parts-based queries).

    Layout:

        <header>

        ---  (divider line)

        Plant 1
        =======
        <full CSV summary for plant 1>

        ---  (divider line)

        Plant 2
        =======
        <full CSV summary for plant 2>

        ...

    This makes it easy to visually separate each plant block in the console.
    """
    chunks: List[str] = []

    header = header.strip()
    if header:
        chunks.append(header)

    for idx, (sci_name, rows) in enumerate(records_by_plant.items(), start=1):
        plant_block = build_answer_from_records(rows).strip()

        # Divider between plants
        chunks.append("\n---\n")
        chunks.append(f"**Plant {idx}**\n\n{plant_block}")

    return "\n".join(chunks)


# ------------------------------
# 9. FOLLOW-UP + MULTI-PLANT HELPERS
# ------------------------------

def is_followup_question(query: str) -> bool:
    """
    Heuristic: short question that refers to "it / this plant / that plant".
    """
    q = f" {query.lower().strip()} "
    if any(p in q for p in [" it ", " its ", " this plant", " that plant"]):
        return len(q.split()) <= 15
    return False


def is_multi_plant_request(query: str) -> bool:
    """
    Detect whether the user is clearly asking about multiple plants
    (e.g. "list all plants that treat cough").
    """
    q = query.lower()

    if "plants" in q:
        return True

    if "plant" in q and any(
        kw in q for kw in ["list", "all", "any", "many", "multiple", "several"]
    ):
        return True

    verbs = ["give", "provide", "show", "fetch", "find", "get", "supply"]
    if "plant" in q and any(v in q for v in verbs):
        return True

    patterns = [
        "what plant in the dataset can treat",
        "which plant in the dataset can treat",
    ]
    return any(p in q for p in patterns)


def focus_on_main_plant(docs):
    """
    If vector retrieval returned docs for multiple plants, try to narrow
    down to a single plant based on the first doc's 'Scientific Name' metadata.
    """
    if not docs:
        return [], None

    main_sci = docs[0].metadata.get("Scientific Name")
    if not main_sci:
        return docs, None

    filtered = [d for d in docs if d.metadata.get("Scientific Name") == main_sci]
    return (filtered or docs), main_sci


# ------------------------------
# 10. LLM (AI MODEL) CONFIGURATION
# ------------------------------

if OllamaLLM is not None and ChatPromptTemplate is not None:
    model = OllamaLLM(model="llama3.2", temperature=0)

    template = """
You are an ethnomedicinal plant assistant for Mindanao, Philippines.

Detected language: {detected_language}
You MUST answer in the same language as the user (English, Tagalog, or Bisaya).

You can ONLY use the text under "DATASET" as your knowledge.
You are NOT allowed to add plants, uses, preparations, or facts that are not clearly supported by the dataset.

IF the answer cannot be found in the dataset, you MUST reply exactly with:

"I'm sorry, but I can't find information about that in the dataset."

and nothing else.

--- Few-shot examples (short) ---
Example (Tagalog):
CONTEXT: [Sambong | Uses: urinary stones]
USER: Para saan ang sambong?
ANSWER: Ayon sa dataset, ginagamit ang Sambong para sa urinary stones. - Source: ExampleStudy (p. 12)

Example (Bisaya):
CONTEXT: [Lagundi | Uses: ubo]
USER: Unsa gamit sa Lagundi?
ANSWER: Sumala sa dataset, gigamit ang Lagundi para sa ubo. - Source: ExampleStudy (p. 11)

CURRENT FOCUS (may be empty if the question is general):
- Active plant (Scientific Name): {active_plant}

ANSWER RULES:
- When listing plant details, use "-" bullets, one fact per line.
- If several different plants in the dataset match, list each plant once (by Scientific Name) and merge their details.
- If the user asks whether a plant can help with a specific condition
  (for example: "Can Tawa-tawa help with my asthma?",
  "Is this plant good for fever?", "Does it treat cough?"), you MUST:
  1) Look only at the diseases/conditions mentioned in the DATASET for that plant.
  2) Start your answer with a clear yes/no sentence:
     - If the condition appears in the dataset for that plant, start with:
       "Yes. According to the dataset, <plant> can be used for <condition>."
     - If the condition does NOT appear anywhere in the dataset for that plant,
       start with:
       "The dataset does not mention <condition> as an indication for <plant>."
  3) After that first sentence, you may add 1–4 bullet points summarizing the
     relevant dataset information (other listed uses, preparation,
     dosage, frequency, and side effects).
- The DATASET may include metadata fields such as "Literature taken from"
  and "Page where the data can be found on the literature". Treat these as the
  official source and page for the information.
- Whenever the user asks about the source, reference, literature, or page
  (for example: "where is White Flower being sourced on", "what is the
  reference", "what page is this from"), you MUST clearly include that
  information in your answer using a format like:
  - "Source: [literature title], page [page number]".
- If multiple matching rows for the same plant have different sources or pages,
  list each distinct "Source + page" combination once.

REASONING RULES:
- Treat the DATASET as the only source of truth.
- Do NOT guess or use outside/world knowledge.
- For every plant, the correct scientific name is already written in the DATASET.
  Whenever you mention a scientific name, you MUST copy it exactly as it appears
  in the DATASET. NEVER substitute it with another scientific name you know
  from outside the dataset (even if you think it is correct).
- Do NOT invent literature titles, authors, or page numbers.
- If the plant name in the question does NOT appear anywhere in the DATASET text,
  treat it as “not in dataset” and use the fallback response.
- If the user asks about a specific condition and that condition does not appear
  anywhere in the DATASET for the plant you are talking about, you MUST follow
  the rule above: clearly say that the dataset does not mention that condition
  for that plant and do not guess new uses.
- If the user specifically asks "where is this data sourced", "what is the
  reference/literature", or "what page is this from", focus your answer on the
  literature and page information for the relevant plant(s).
- If the question is not about ethnomedicinal plants in Mindanao, briefly refuse
  and say your scope is limited.

====================
CHAT HISTORY
{history}
====================
DATASET
{reviews}
====================
USER QUESTION
{question}

YOUR ANSWER:
"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
else:
    model = None
    chain = None


def is_fallback_message(text: str) -> bool:
    """
    Detect if a text is essentially the dataset fallback message.
    """
    if not text:
        return False

    cleaned = str(text).strip().replace("’", "'").lower()
    core = "can't find information about that in the dataset"
    return core in cleaned


# ------------------------------
# 11. MAIN CHAT LOOP
# ------------------------------

def main() -> None:
    """
    Entry point for the ethnomedicinal plant chatbot.

    This version:
      - Uses CSV-based logic as much as possible (no LLM).
      - Supports English, Tagalog, and Bisaya questions.
      - Recognizes both scientific and local plant names using the dataset.
    """

    console.print(
        Panel(
            "🌿 Arcana Herbarium – Mindanao Plant Codex",
            style="bold green",
        )
    )

    history: List[str] = []
    last_reviews = None
    last_plant_sci = None

    while True:
        query = console.input(
            "[bold cyan]Inscribe your question into the Arcana Herbarium (press q to close the codex): [/bold cyan]"
        )

        if query.strip().lower() == "q":
            console.print(
                "[bold yellow]Closing the codex... may your remedies be effective and your plants well-tended. 🌿[/bold yellow]"
            )
            break

        lower_q = query.lower().replace("’", "'")

        # Detect one or more plant names directly from the question text.
        plants_in_text = find_all_plants_in_text(query)
        plant_in_text = plants_in_text[0] if plants_in_text else None

        # --------------------------------------------------------
        # 1. "Prep method" questions – CSV only (English heuristic)
        # --------------------------------------------------------
        plant_for_prep: Optional[str] = None
        prep_conditions: List[str] = []

        if (
            ("prep method" in lower_q or "preparation method" in lower_q or "preparation" in lower_q)
            and ("to treat" in lower_q or " for " in lower_q)
        ):
            plant_for_prep = plant_in_text

            if not plant_for_prep and last_plant_sci and is_followup_question(query):
                plant_for_prep = last_plant_sci

            if plant_for_prep:
                raw_conditions = extract_condition_phrases_from_query(query)

                if is_condition_actually_disease(plant_for_prep, raw_conditions):
                    prep_conditions = raw_conditions
                else:
                    prep_conditions = []

        if plant_for_prep:
            answer = build_prep_method_answer(plant_for_prep, prep_conditions)

            if is_fallback_message(answer):
                print_error(answer)
            else:
                print_answer(answer)

            history.append(f"User: {query}")
            history.append(f"Assistant: {answer}")

            last_plant_sci = plant_for_prep
            last_reviews = retrieve_with_threshold(plant_for_prep)
            continue

        # --------------------------------------------------------
        # 2. Condition-specific YES/NO questions (English)
        # --------------------------------------------------------
        plant_for_condition: Optional[str] = None
        condition_phrases: List[str] = []

        if is_condition_question(query):
            plant_fragment, cond_fragment = parse_is_used_for_question(query)

            if plant_fragment and cond_fragment:
                plant_for_condition = find_plant_in_text(plant_fragment)
                condition_phrases = extract_condition_phrases_from_query(cond_fragment)

        # Follow-up style: "Can it treat cough?"
        if (not plant_for_condition) and last_plant_sci and is_followup_question(query):
            plant_for_condition = last_plant_sci
            condition_phrases = extract_condition_phrases_from_query(query)

        if plant_for_condition and condition_phrases:
            answer = build_condition_answer(plant_for_condition, condition_phrases)

            if is_fallback_message(answer):
                print_error(answer)
            else:
                print_answer(answer)

            history.append(f"User: {query}")
            history.append(f"Assistant: {answer}")

            last_plant_sci = plant_for_condition
            last_reviews = retrieve_with_threshold(plant_for_condition)
            continue

        # --------------------------------------------------------
        # 3. Single-plant CSV answers (any language)
        #    e.g. "What is white flower?", "Ano ang gamit ng Lagundi?",
        #         "Unsaon pag gamit sa tanglad para ubo?"
        # --------------------------------------------------------
        if plants_in_text:
            # Only one plant mentioned → behave like before
            if len(plants_in_text) == 1:
                target_sci = plants_in_text[0]
                records = get_raw_records_for_name(target_sci)
                if records:
                    answer = build_intro_answer_from_records(records)

                    print_answer(answer)

                    history.append(f"User: {query}")
                    history.append(f"Assistant: {answer}")

                    last_plant_sci = str(records[0]["Scientific Name"])
                    last_reviews = retrieve_with_threshold(last_plant_sci)
                    continue
            else:
                # Multiple plants mentioned → show each one in a separate block
                blocks: List[str] = []
                for idx, sci_name in enumerate(plants_in_text, start=1):
                    records = get_raw_records_for_name(sci_name)
                    if not records:
                        continue
                    block = build_intro_answer_from_records(records)
                    blocks.append(f"---\n**Plant {idx}**\n\n{block}")

                if blocks:
                    final_answer = "\n\n".join(blocks)

                    print_answer(final_answer)

                    history.append(f"User: {query}")
                    history.append(f"Assistant: {final_answer}")

                    # No single "last plant" for follow-up like "what about its dosage?"
                    last_plant_sci = None
                    last_reviews = None
                    continue

        # --------------------------------------------------------
        # 4. Multi-plant CSV search:
        #    "list all plants that treat jaundice"
        # --------------------------------------------------------
        multi_request = is_multi_plant_request(query)

        if multi_request:
            # 4A. PARTS-based multi-plant search
            if is_parts_based_request(query):
                part_keywords = extract_parts_keywords_from_query(query)
                if not part_keywords:
                    part_keywords = ["leaf", "leaves", "lf"]

                records = search_records_by_parts_keywords(part_keywords)

                if records:
                    records_by_plant: Dict[str, List[Dict]] = {}
                    for row in records:
                        sci_name = str(row["Scientific Name"])
                        records_by_plant.setdefault(sci_name, []).append(row)

                    num_plants = len(records_by_plant)
                    unique_parts = sorted({kw.lower() for kw in part_keywords})
                    parts_text = ", ".join(unique_parts)
                    header = (
                        f"**{num_plants} matching plants found for parts:** "
                        f"{parts_text}"
                    )

                    final_answer = build_multi_plant_grouped_answer(records_by_plant, header)

                    # Highlight part terms (leaf, root, etc.)
                    highlight_terms = set()
                    for kw in part_keywords:
                        kw = kw.strip()
                        if not kw:
                            continue
                        highlight_terms.add(kw)
                        for piece in re.split(r"[;,/ ]+", kw):
                            piece = piece.strip()
                            if len(piece) >= 3:
                                highlight_terms.add(piece)

                    final_answer = highlight_keywords(final_answer, list(highlight_terms))

                    print_answer(final_answer)

                    history.append(f"User: {query}")
                    history.append(f"Assistant: {final_answer}")

                    last_plant_sci = None
                    last_reviews = None
                    continue
                # else: fall through to disease-based search

            # 4B. DISEASE-based multi-plant search
            disease_keywords = extract_condition_phrases_from_query(query)
            if not disease_keywords:
                disease_keywords = ["cough", "colds"]

            records = search_records_by_disease_keywords(disease_keywords)

            if records:
                records_by_plant: Dict[str, List[Dict]] = {}
                for row in records:
                    sci_name = str(row["Scientific Name"])
                    records_by_plant.setdefault(sci_name, []).append(row)

                num_plants = len(records_by_plant)
                unique_conditions = sorted({kw.lower() for kw in disease_keywords})
                conditions_text = ", ".join(unique_conditions)
                header = (
                    f"**{num_plants} matching plants found for:** "
                    f"{conditions_text}"
                )

                final_answer = build_multi_plant_grouped_answer(records_by_plant, header)

                highlight_terms = set()
                for kw in disease_keywords:
                    kw = kw.strip()
                    if not kw:
                        continue
                    highlight_terms.add(kw)
                    for piece in re.split(r"[;,/ ]+", kw):
                        piece = piece.strip()
                        if len(piece) >= 4:
                            highlight_terms.add(piece)

                final_answer = highlight_keywords(final_answer, list(highlight_terms))

                print_answer(final_answer)

                history.append(f"User: {query}")
                history.append(f"Assistant: {final_answer}")

                last_plant_sci = None
                last_reviews = None
                continue

        # --------------------------------------------------------
        # 5. Fallback: vector retrieval + LLM (dataset-only)
        # --------------------------------------------------------
        if chain is None:
            # LLM not available
            print_error(
                "LLM backend is not available. I can only answer using the "
                "direct CSV rules right now."
            )
            continue

        # Follow-up questions referring to the last plant: use its docs
        if is_followup_question(query) and last_plant_sci is not None and last_reviews is not None:
            reviews = last_reviews
            active_plant = last_plant_sci
        else:
            raw_docs = retrieve_with_threshold(query)
            if not raw_docs:
                print_error(
                    "No matching plant information found in the dataset.\n"
                    "This might mean the plant is not in the dataset, or the spelling is very different."
                )
                continue

            focused_docs, plant_sci = focus_on_main_plant(raw_docs)
            reviews = focused_docs
            last_reviews = focused_docs
            last_plant_sci = plant_sci
            active_plant = plant_sci if plant_sci else "Not specified"

        history_text = "\n".join(history[-8:])

        detected_language = detect_language(query)

        result = chain.invoke(
            {
                "active_plant": active_plant,
                "history": history_text,
                "reviews": reviews,
                "question": query,
                "detected_language": detected_language,
            }
        )

        result = tidy_response(result)

        if is_fallback_message(result):
            print_error(result)
        else:
            print_answer(result)

        history.append(f"User: {query}")
        history.append(f"Assistant: {result}")


if __name__ == "__main__":
    main()
