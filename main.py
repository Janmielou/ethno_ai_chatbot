# main.py
# ============================================================
# Ethnomedicinal Plant Chatbot (Mindanao, Philippines)
#
# Console-based chatbot that answers questions about
# ethnomedicinal plants found in Mindanao, Philippines.
#
# Core behavior:
#   1. Load plant data from a CSV via helper functions in vector.py.
#   2. Try to answer from the CSV directly (no AI) whenever possible:
#        - Single plant questions  → direct CSV summary.
#        - Multi-plant questions   → search CSV by disease/parts.
#        - Condition questions     → YES/NO answers per plant.
#        - Prep-method questions   → condition-specific prep details.
#   3. Only when needed, call a local LLM (LLaMA 3.2 via Ollama)
#      to answer using strictly the dataset text (no outside info).
#   4. Display everything nicely using Rich panels in the terminal.
# ============================================================

# ------------------------------
# 1. IMPORTS
# ------------------------------

from langchain_ollama.llms import OllamaLLM           # Connects to local LLaMA 3.2 via Ollama.
from langchain_core.prompts import ChatPromptTemplate  # Helps build prompts with placeholders.

from vector import (
    retrieve_with_threshold,              # Vector search over CSV text chunks.
    get_known_plant_names,                # All scientific names in the dataset.
    get_raw_records_for_name,             # All CSV rows for a given scientific name.
    search_records_by_disease_keywords,   # CSV search by "Disease used on".
    search_records_by_parts_keywords,     # CSV search by "Parts used".
    get_all_disease_phrases,              # All distinct "Disease used on" phrases.
    get_dataset_vocabulary,               # All tokens appearing in the dataset.
)

import re  # Regular expressions for pattern matching.
import difflib

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

# Rich console (for pretty printing)
console = Console()


# ------------------------------
# 2. DATASET-DERIVED CONSTANTS
# ------------------------------

# All scientific plant names present in the CSV.
KNOWN_PLANT_NAMES = get_known_plant_names()

# All phrases from the "Disease used on" column.
ALL_DISEASE_PHRASES = get_all_disease_phrases()

# All distinct tokens appearing anywhere in the CSV;
# used to detect questions that are "out of scope".
DATASET_VOCAB = get_dataset_vocabulary()

# ------------------------------------------------------------
# FUZZY SPELL-FIX FOR USER QUERIES
# ------------------------------------------------------------

# Extra words we always want the spellchecker to know about,
# even if they don't appear directly in the dataset vocabulary.
CORE_EXTRA_SPELLFIX_WORDS: set[str] = {
    # Generic helpers
    "what", "which", "who", "where", "when", "why", "how",
    "can", "could", "would", "should", "may", "might",
    "is", "are", "was", "were", "do", "does", "did",
    "plant", "plants", "used", "use", "treat", "treats",
    "for", "from", "with", "part", "parts", "prep", "list",

    # Domain / thesis words we care about
    "preparation", "preparations",
    "method", "methods",
    "administration", "frequency", "frequencies",
    "dosage", "dose", "doses",
    "medicinal", "medicinally",
    "ethnomedicinal",
}

# This will hold all tokens we consider "valid" for spell-fixing.
SPELLFIX_VOCAB: set[str] = set()


def _build_spellfix_vocab() -> None:
    """
    Populate SPELLFIX_VOCAB from:
      - all tokens that appear anywhere in the dataset
      - CORE_EXTRA_SPELLFIX_WORDS above
    """
    global SPELLFIX_VOCAB
    SPELLFIX_VOCAB.clear()

    # 1) Tokens from dataset vocabulary (phrases → tokens)
    for w in DATASET_VOCAB:
        w = str(w).strip().lower()
        if not w:
            continue
        for token in re.split(r"[^a-z]+", w):
            token = token.strip()
            if token:
                SPELLFIX_VOCAB.add(token)

    # 2) Plus our hand-picked extras
    SPELLFIX_VOCAB.update(CORE_EXTRA_SPELLFIX_WORDS)


_build_spellfix_vocab()  # build it once at startup


def _closest_spellfix_word(token: str) -> str | None:
    """
    Return the closest known word to `token` using fuzzy matching,
    or None if nothing looks close enough.
    """
    token = token.lower()
    if token in SPELLFIX_VOCAB or len(token) < 3:
        return None

    matches = difflib.get_close_matches(token, SPELLFIX_VOCAB, n=1, cutoff=0.80)
    return matches[0] if matches else None


def normalize_query_spelling(raw_query: str) -> tuple[str, bool]:
    """
    Light fuzzy spell-fix for user queries.

    Returns:
        (fixed_query, changed_flag)

    - fixed_query: the possibly corrected text
    - changed_flag: True if we actually changed anything
    """
    # Pre-compute tokens that belong to known scientific names / aliases
    known_name_tokens: set[str] = set()
    for sci in KNOWN_PLANT_NAMES:
        for t in re.split(r"[^a-z]+", str(sci).lower()):
            if len(t) >= 3:
                known_name_tokens.add(t)

    for alias in NAME_TO_SCI.keys():
        for t in re.split(r"[^a-z]+", str(alias).lower()):
            if len(t) >= 3:
                known_name_tokens.add(t)

    # Split but KEEP separators (spaces, punctuation) so we can rebuild string
    parts = re.split(r"(\W+)", raw_query)
    fixed_parts: list[str] = []

    for part in parts:
        # Non-alphabetic chunks (spaces, punctuation) → keep
        if not part or not any(ch.isalpha() for ch in part):
            fixed_parts.append(part)
            continue

        lower = part.lower()

        # Very short tokens → leave them alone
        if len(lower) <= 2:
            fixed_parts.append(part)
            continue

        # Don't touch pieces of plant names (peperomia, pellucida, etc.)
        if lower in known_name_tokens:
            fixed_parts.append(part)
            continue

        # Already known word → keep
        if lower in SPELLFIX_VOCAB:
            fixed_parts.append(part)
            continue

        # Try fuzzy correction
        replacement = _closest_spellfix_word(lower)
        if not replacement:
            fixed_parts.append(part)
            continue

        # Preserve capitalization of the original token
        if part[0].isupper():
            replacement = replacement.capitalize()

        fixed_parts.append(replacement)

    fixed_query = "".join(fixed_parts)
    changed = fixed_query != raw_query
    return fixed_query, changed


# Generic words we ignore when deciding if a query is out-of-scope.
SCOPE_STOPWORDS = {
    "what", "which", "who", "where", "when", "why", "how",
    "is", "are", "was", "were", "be", "being", "been",
    "do", "does", "did",
    "in", "on", "at", "for", "from", "to", "of", "by", "with",
    "about", "as", "into", "through", "over", "under",
    "the", "a", "an", "this", "that", "these", "those",
    "my", "your", "their", "our",
    "plant", "plants", "tree", "trees", "herb", "herbs",
    "used", "use", "uses", "using",
    "treat", "treats", "treating", "treatment",
    "can", "could", "would", "should", "may", "might",
    "dataset", "record", "records", "data", "information",
    "please", "kindly", "tell", "explain", "give", "show", "list",
    "help", "helps", "helping",
    "cure", "cures", "curing",
    "provide", "provides", "providing",
    "fetch", "fetches", "fetching",
    "get", "gets", "getting",
    "find", "finds", "finding",
    "all", "dosage", "dose", "doses", "frequency",
    "frequencies",
}

# Stopwords used when extracting disease/condition phrases from questions.
DISEASE_STOPWORDS = {
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

# Build a set of allowed genera.
# This helps detect if the LLM invents a "new" scientific name
# using an existing genus but a species that isn't in the dataset.
ALLOWED_GENERA: set[str] = set()
for full in KNOWN_PLANT_NAMES:
    parts = str(full).split()
    if not parts:
        continue
    first = "".join(ch for ch in parts[0] if ch.isalpha())
    if first:
        ALLOWED_GENERA.add(first.lower())


# ------------------------------
# 3. PLANT SYNONYMS & NAME INDEX
# ------------------------------

# Mapping from canonical scientific name → list of aliases / local names.
# The canonical key MUST exist in KNOWN_PLANT_NAMES.
PLANT_SYNONYMS: dict[str, list[str]] = {
    # EXAMPLE: Andrographis paniculata ("White flower")
    "Andrographis paniculata (Burm.f.) Nees": [
        "andrographis paniculata",
        "white flower",
        "whiteflower",
        "white-flower",
        "white flowers",
        "WHITE FLOWER",
    ],

    # Tawa-tawa ↔ Euphorbia hirta L.
    "Euphorbia hirta L.": [
        "euphorbia hirta",
        "tawa-tawa",
        "tawa tawa",
        "tawatawa",
        "tawa tawaa",
        "tawa tawa plant",
        "TAWA TAWA",
    ],

    "Lansium domesticum Correa": [
        "lansium domesticum",
        "lansones",
        "Lansones",
    ],

    # 🔹 ADD THIS BLOCK FOR LAGUNDI 🔹
    "Vitex negundo L.": [
        "lagundi",
        "Lagundi",
        "vitex negundo",
        "five-leaved chaste tree",
    ],

    # ... rest of your plants ...
}


# NAME_TO_SCI: every known string (scientific name or alias) → canonical scientific name.
NAME_TO_SCI: dict[str, str] = {}


def _build_name_index() -> None:
    """
    Fill NAME_TO_SCI so that any known name or synonym maps
    back to a canonical scientific name from the dataset.
    """
    global NAME_TO_SCI
    NAME_TO_SCI.clear()

    # 1) From PLANT_SYNONYMS
    for sci_name, syns in PLANT_SYNONYMS.items():
        sci_key = sci_name.strip().lower()
        if sci_key:
            NAME_TO_SCI[sci_key] = sci_name  # map scientific name to itself

        for s in syns:
            key = s.strip().lower()
            if key:
                NAME_TO_SCI[key] = sci_name

    # 2) From the dataset scientific names themselves
    for sci in KNOWN_PLANT_NAMES:
        key = str(sci).strip().lower()
        if key and key not in NAME_TO_SCI:
            NAME_TO_SCI[key] = str(sci)


def normalize_query_spelling(text: str) -> tuple[str, bool]:
    """
    Very small, dataset-aware spell fixer.

    For each alphabetic token that is NOT in SPELLFIX_VOCAB:
      - Find a close match inside SPELLFIX_VOCAB using difflib.
      - If the similarity is high enough (cutoff), replace it.

    This automatically handles things like:
      - 'waht'        -> 'what'
      - 'sued'        -> 'used'
      - 'preperation' -> 'preparation'
      - 'emthod'      -> 'method'
      - 'medicnally'  -> 'medicinally'
    as long as the correct word exists in the dataset or CORE_EXTRA_SPELLFIX_WORDS.

    Returns:
      (possibly_corrected_text, did_change_flag)
    """
    tokens = re.split(r"(\W+)", text)  # keep punctuation as separate tokens
    changed = False
    new_tokens: list[str] = []

    # Pre-materialize vocab list once for difflib
    vocab_list = list(SPELLFIX_VOCAB)

    for tok in tokens:
        # Only care about pure alphabetic tokens; keep numbers / punctuation as is.
        if tok.isalpha():
            lower = tok.lower()

            # Already a known word → leave it alone.
            if lower in SPELLFIX_VOCAB:
                new_tokens.append(tok)
                continue

            length = len(lower)
            if length < 3:
                # too short, not worth correcting
                new_tokens.append(tok)
                continue

            # Slightly stricter for longer words so we don't over-correct.
            if length <= 5:
                cutoff = 0.75
            else:
                cutoff = 0.80

            # Find the closest match in the vocab.
            match = difflib.get_close_matches(
                lower, vocab_list, n=1, cutoff=cutoff
            )

            if match:
                fixed = match[0]
                # Preserve capitalization if the user capitalized the word.
                if tok[0].isupper():
                    fixed = fixed.capitalize()
                new_tokens.append(fixed)
                if fixed != lower:
                    changed = True
                continue  # move on to next token

        # Default: keep token unchanged.
        new_tokens.append(tok)

    return "".join(new_tokens), changed


def _normalize_plant_string_for_match(s: str) -> str:
    """
    Normalize text for plant-name comparison (currently not used,
    but kept for possible future use).
    """
    s = s.lower().replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def find_plant_in_text(text: str) -> str | None:
    """
    Scan free text and return the *scientific name* if we recognize
    any plant name / synonym inside it.
    """
    if not text:
        return None

    t = text.lower()
    best_match: str | None = None
    best_len = 0

    for phrase, sci_name in NAME_TO_SCI.items():
        if phrase and phrase in t and len(phrase) > best_len:
            best_len = len(phrase)
            best_match = sci_name

    return best_match


# Build the index at startup.
_build_name_index()


# ------------------------------
# 4. LLM (AI MODEL) CONFIGURATION
# ------------------------------

# Local LLaMA 3.2 model accessed through Ollama.
# temperature=0 makes it deterministic.
model = OllamaLLM(model="llama3.2", temperature=0)

# System template that strictly constrains the LLM to dataset-only knowledge.
template = """
You are an ethnomedicinal plant assistant for Mindanao, Philippines.

You can ONLY use the text under "DATASET" as your knowledge.
You are NOT allowed to add plants, uses, preparations, or facts that are not clearly supported by the dataset.

IF the answer cannot be found in the dataset, you MUST reply exactly with:

"I'm sorry, but I can't find information about that in the dataset."

and nothing else.

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

# Standard text when the dataset cannot answer.
FALLBACK_MSG = "I'm sorry, but I can't find information about that in the dataset."


# ------------------------------
# 5. GENERIC HELPERS
# ------------------------------

def print_answer(text: str) -> None:
    """Show a successful answer in a green Rich panel."""
    panel = Panel(Markdown(text), title="🌿 ANSWER", border_style="green", padding=(1, 2))
    console.print(panel)


def print_error(text: str) -> None:
    """Show an error or warning in a red Rich panel."""
    panel = Panel(text, title="⚠️ ERROR", border_style="red", padding=(1, 2))
    console.print(panel)


def tidy_response(text: str) -> str:
    """
    Lightly reformat model output so bullets look nice in the terminal.
    """
    text = text.replace(" -", "\n-")
    text = text.replace("•", "\n•")
    return text.strip()


def highlight_keywords(text: str, keywords: list[str]) -> str:
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


def query_has_out_of_scope_keyword(query: str) -> bool:
    """
    Return True if the question contains several words that NEVER
    appear in the dataset, suggesting it's off-topic.
    """
    tokens = re.split(r"[^\w]+", query.lower())
    unknown = 0
    for tok in tokens:
        tok = tok.strip()
        if len(tok) < 3:
            continue
        if tok in SCOPE_STOPWORDS:
            continue
        if tok not in DATASET_VOCAB:
            unknown += 1
            if unknown >= 3:
                return True
    return False


def is_fallback_message(text: str) -> bool:
    """
    Detect if a text is essentially the dataset fallback message.
    """
    if not text:
        return False

    cleaned = str(text).strip().replace("’", "'").lower()
    core = "can't find information about that in the dataset"
    return core in cleaned


def contains_unknown_scientific_name(answer: str) -> bool:
    """
    Check if the model invented a scientific name that isn't present
    in KNOWN_PLANT_NAMES, using any known genus as a clue.
    """
    pattern = r"\b([A-Z][a-z]+)\s+([a-z\-]+)\b"

    for match in re.finditer(pattern, answer):
        genus = match.group(1).lower()
        species = match.group(2).lower()

        if genus not in ALLOWED_GENERA:
            continue

        candidate = f"{genus} {species}"
        if not any(candidate in name for name in KNOWN_PLANT_NAMES):
            return True

    return False


def is_followup_question(query: str) -> bool:
    """
    Heuristic: short question that refers to "it / this plant / that plant".
    """
    q = f" {query.lower().strip()} "
    if any(p in q for p in [" it ", " its ", " this plant", " that plant"]):
        return len(q.split()) <= 15
    return False


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


def extract_single_plant_name(query: str) -> str | None:
    """
    Try to extract a single plant fragment from 'what is X' style questions.
    """
    q = query.strip().lower().replace("’", "'")

    if q and q[-1] in "?.!":
        q = q[:-1].strip()

    prefixes = [
        "what is ",
        "what's ",
        "whats ",
        "what about ",
        "tell me about ",
        "info about ",
        "information about ",
        "what can you tell me about ",
        "what disease is ",
        "what diseases is ",
        "what disease are ",
        "what diseases are ",
        "what is the local name of ",
        "what's the local name of ",
        "whats the local name of ",
    ]

    for p in prefixes:
        if q.startswith(p):
            name = q[len(p):].strip()
            return name or None

    return None

def looks_like_plant_question(text: str) -> bool:
    """
    Heuristic: return True only if the query *looks* related to plants /
    ethnomedicine. Otherwise we skip auto-correction.

    This avoids 'fixing' totally unrelated or gibberish questions.
    """
    if not text:
        return False

    t = text.lower()

    # If we can detect a known plant name or synonym → definitely plant-related
    if find_plant_in_text(text):
        return True

    # Simple keyword hints that the question is about plants / uses / ailments
    HINT_WORDS = [
        "plant", "leaf", "leaves", "root", "bark", "herb",
        "parts used", "disease", "used for", "treat", "treats",
        "cough", "fever", "colds", "asthma", "wound", "skin", "burn",
        "dosage", "medicinal", "ethnomedicinal",
    ]

    return any(hint in t for hint in HINT_WORDS)


def apply_spellfix(raw_query: str) -> tuple[str, bool]:
    """
    Very conservative spell-fix.

    - Only runs if the question looks plant-related.
    - Never changes plant names (scientific or local).
    - Only fixes obvious typos where the suggested word:
        * is in SPELLFIX_VOCAB
        * has the same first letter
        * length difference <= 1
    """
    # If the question doesn't even look like it's about plants / ailments,
    # don't try to "fix" anything.
    if not looks_like_plant_question(raw_query):
        return raw_query, False

    parts = re.split(r"(\W+)", raw_query)
    fixed_parts: list[str] = []
    changed = False
    vocab_list = list(SPELLFIX_VOCAB)

    # Build a set of tokens that belong to plant names / aliases
    plant_tokens: set[str] = set()
    for sci in KNOWN_PLANT_NAMES:
        for t in re.split(r"[^a-z]+", str(sci).lower()):
            if len(t) >= 3:
                plant_tokens.add(t)
    for alias in NAME_TO_SCI.keys():
        for t in re.split(r"[^a-z]+", str(alias).lower()):
            if len(t) >= 3:
                plant_tokens.add(t)

    for part in parts:
        # Keep spaces/punctuation as-is
        if not part or not any(ch.isalpha() for ch in part):
            fixed_parts.append(part)
            continue

        lower = part.lower()

        # Very short tokens, already-known words, or plant tokens → don't touch
        if (
            len(lower) <= 2
            or lower in SPELLFIX_VOCAB
            or lower in plant_tokens
        ):
            fixed_parts.append(part)
            continue

        # Ask difflib for a close match
        match = difflib.get_close_matches(lower, vocab_list, n=1, cutoff=0.80)
        if not match:
            fixed_parts.append(part)
            continue

        candidate = match[0]

        # SAFETY CHECKS:
        #  - don't drastically shrink/extend the word
        #  - force same first letter (prevents "autism" → "tism")
        if abs(len(candidate) - len(lower)) > 1 or candidate[0] != lower[0]:
            fixed_parts.append(part)
            continue

        replacement = candidate
        # Preserve capitalization
        if part[0].isupper():
            replacement = replacement.capitalize()

        fixed_parts.append(replacement)
        if replacement != lower:
            changed = True

    fixed_query = "".join(fixed_parts)
    return fixed_query, changed



# ------------------------------
# 6. CONDITION / DISEASE HELPERS
# ------------------------------

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


def parse_is_used_for_question(query: str) -> tuple[str | None, str | None]:
    """
    Parse questions like:
      'Is Andrographis paniculata used for cough?'
    into (plant_fragment, condition_fragment).
    """
    q = query.lower().replace("’", "'").strip()

    m = re.search(
        r"\b(?:is|are)\s+(.+?)\s+"
        r"(?:used for|good for|for treating|to treat|treat(?:ing)?|treats?)\s+"
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


def extract_condition_phrases_from_query(query: str) -> list[str]:
    """
    Try to pull out the *condition phrase(s)* from the user's question.

    1) Prefer things after 'to treat X', 'treat X', 'helps with X'.
    2) Otherwise fall back to 'used for X', 'for X', 'as X'.
    3) If that still fails, return keyword tokens.
    """
    import re

    # Same stopword list you were using before
    DISEASE_STOPWORDS = {
        "what", "which", "who", "where", "when", "why", "how",
        "is", "are", "was", "were", "be", "being", "been",
        "do", "does", "did",
        "can", "could", "would", "should", "may", "might",
        "in", "on", "at", "for", "from", "to", "of", "by", "with",
        "the", "a", "an", "this", "that", "these", "those",
        "plant", "plants", "tree", "trees", "herb", "herbs",
        "treat", "treats", "treating", "treatment",
        "cure", "cures", "curing",
        "heal", "heals", "healing",
        "relief", "relieve", "relieves",
        "used", "use", "uses", "using",
        "give", "show", "list", "provide", "fetch",
        "please", "kindly",
        "pain", "pains", "ache", "aches",
        "all", "any", "some", "as",
        "medicinal", "part", "parts",
        "disease", "diseases",
        "condition", "conditions",
        "problem", "problems",
    }

    q = query.lower()
    q = q.replace("’", "'")

    # 1) First, check explicit multi-word phrases if you like
    MULTIWORD_CONDITIONS = [
        "cancer therapeutics",
        "gas pain",
        "stomach pain",
        "high blood pressure",
        "high blood",
    ]
    found_multi: list[str] = []
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

def _get_plant_tokens(plant_name: str) -> set[str]:
    """
    Collect tokens that belong to a plant's scientific name + all its
    synonyms/aliases. Used for detecting and trimming plant words from
    condition phrases.
    """
    plant_tokens: set[str] = set()

    # Tokens from the canonical scientific name
    for t in re.split(r"[^a-z]+", plant_name.lower()):
        if len(t) >= 3:
            plant_tokens.add(t)

    # Tokens from PLANT_SYNONYMS + NAME_TO_SCI aliases
    synonyms: set[str] = set(PLANT_SYNONYMS.get(plant_name, []))
    for alias, sci in NAME_TO_SCI.items():
        if sci == plant_name:
            synonyms.add(alias)

    for s in synonyms:
        for t in re.split(r"[^a-z]+", s.lower()):
            if len(t) >= 3:
                plant_tokens.add(t)

    return plant_tokens


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


def is_condition_actually_disease(plant_name: str, condition_phrases: list[str]) -> bool:
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
    condition_tokens: set[str] = set()
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





# Synonym groups for disease phrases:
# user phrase → list of CSV-style keywords.
SYNONYM_GROUPS: dict[str, list[str]] = {
    "stomach pain": [
        "stomachache",
        "stomach trouble",
        "stomach problem",
        "stomach acidity",
        "abdominal pain",
    ],
    "tummy ache": [
        "stomachache",
        "abdominal pain",
        "stomach trouble",
    ],
    "tummy pain": [
        "stomachache",
        "abdominal pain",
        "stomach trouble",
    ],
    "high fever": [
        "fever",
    ],
    "cough with phlegm": [
        "cough with phlegm",
    ],
    "cold and chills": [
        "colds",
        "body chills",
        "fever",
    ],
}


def extract_disease_keywords_from_query(query: str) -> list[str]:
    """
    Convert a natural-language disease question into keywords that match
    the 'Disease used on' column.
    """
    q = query.lower()

    # 0) Multi word synonyms like "stomach pain"
    for user_phrase, target_keywords in SYNONYM_GROUPS.items():
        if user_phrase in q:
            return target_keywords

    matched_phrases: list[str] = []

    # 1) Try to use actual phrases in the dataset.
    for phrase in ALL_DISEASE_PHRASES:
        phrase_lower = phrase.lower()
        words = [
            w for w in phrase_lower.split()
            if w and w not in DISEASE_STOPWORDS
        ]
        if not words:
            continue
        if all(w in q for w in words):
            matched_phrases.append(phrase)

    if matched_phrases:
        return matched_phrases

    # 2) Fallback: word-based extraction.
    cleaned = q
    for ch in "?,.!":
        cleaned = cleaned.replace(ch, " ")
    tokens = [
        t for t in cleaned.split()
        if t and t not in DISEASE_STOPWORDS
    ]
    return tokens


def is_condition_question(query: str) -> bool:
    """
    True for yes/no style condition questions:
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

PART_TERMS = [
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


def is_parts_based_request(query: str) -> bool:
    """
    Return True if the question is clearly about plant parts.
    """
    q = query.lower()
    if "part " in q or "parts " in q or "parts used" in q:
        return True
    return any(term in q for term in PART_TERMS)


def extract_parts_keywords_from_query(query: str) -> list[str]:
    """
    From a question like:
      'List plants that use leaves as medicinal parts'
    return keywords that match the CSV "Parts used" column.
    """
    q = query.lower()
    found: set[str] = set()

    PART_SYNONYMS = {
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

def _clean_field(val) -> str:
    """Utility: treat NaN/None as empty and normalize whitespace."""
    s = str(val)
    if s.lower() == "nan":
        return ""
    return re.sub(r"\s+", " ", s).strip()


def build_answer_from_records(records: list[dict]) -> str:
    """
    Plain CSV summary for a plant (or a set of rows for that plant).
    """
    if not records:
        return FALLBACK_MSG

    def collect_unique(key: str) -> list[str]:
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

    def join_many(values: list[str]) -> str:
        return values[0] if len(values) == 1 else "; ".join(values)

    title = local_main or "Unknown local name"
    if sci_main:
        title = f"{title} ({sci_main})"

    lines: list[str] = [f"**{title}**"]

    if families:
        lines.append(f"- Family: {join_many(families)}")
    if parts_list:
        lines.append(f"- Parts used: {join_many(parts_list)}")
    if diseases:
        lines.append(f"- Uses: {join_many(diseases)}")
    if preps:
        lines.append(f"- Preparation and administration: {join_many(preps)}")
    if dosages:
        lines.append(f"- Quality of dosage: {join_many(dosages)}")
    if freqs:
        lines.append(f"- Administration frequency: {join_many(freqs)}")
    if sides:
        lines.append(f"- Reported side effects: {join_many(sides)}")

    # Sources: (Literature, Page)
    sources = set()
    for row in records:
        lit = _clean_field(row.get("Literature taken from", ""))
        page = _clean_field(row.get("Page where the data can be found on the literature", ""))
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


def build_usage_summary_answer(plant_name: str) -> str:
    """
    For “What disease is X used for?” questions.
    Intro sentence summarizing all uses, plus full details.
    """
    records = get_raw_records_for_name(plant_name)
    if not records:
        return FALLBACK_MSG

    def collect_unique(key: str) -> list[str]:
        seen, values = set(), []
        for row in records:
            v = _clean_field(row.get(key, ""))
            if v and v not in seen:
                seen.add(v)
                values.append(v)
        return values

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

    diseases = collect_unique("Disease used on")

    if diseases:
        disease_list = "; ".join(diseases)
        first_line = (
            f"{plant_label} can be used to treat the following diseases "
            f"or conditions: {disease_list}."
        )
    else:
        first_line = (
            f"{plant_label} is present in the dataset, but no specific "
            f"diseases or indications are listed."
        )

    detailed = build_answer_from_records(records)
    return first_line + "\n\n" + detailed


def build_condition_answer(plant_name: str, condition_phrases: list[str]) -> str:
    """
    YES/NO condition-specific answer for a single plant.
    """
    cleaned_conditions: list[str] = []
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

    def row_mentions_condition(row: dict, cond: str) -> bool:
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

    filtered: list[dict] = []
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
    def collect_unique(rows: list[dict], key: str) -> list[str]:
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

    def join_many(values: list[str]) -> str:
        return values[0] if len(values) == 1 else "; ".join(values)

    lines: list[str] = []
    lines.append(
        f"Yes. According to the dataset, {plant_label} can be used for **{main_condition}**."
    )
    lines.append("")

    if diseases:
        lines.append(f"- Disease used on: {join_many(diseases)}")
    if parts_list:
        lines.append(f"- Parts used (for **{main_condition}**): {join_many(parts_list)}")
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
        page = _clean_field(row.get("Page where the data can be found on the literature", ""))
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


def build_intro_answer_from_records(records: list[dict]) -> str:
    """
    For "What is X?" questions:
      - Short descriptive paragraph.
      - Detailed dataset-based bullet list.
    """
    if not records:
        return FALLBACK_MSG

    def collect_unique(key: str) -> list[str]:
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

    # Intro paragraph
    sentence_parts: list[str] = []
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
        parts_text = parts[0] if len(parts) == 1 else "; ".join(parts)
        sentence_parts.append(f"The recorded medicinal parts used include {parts_text}.")

    intro_paragraph = " ".join(sentence_parts)

    # Detailed block
    lines: list[str] = ["**Detailed information from the dataset:**"]
    if local_main:
        lines.append(f"- Local Name: {local_main}")
    if sci_main:
        lines.append(f"- Scientific Name: {sci_main}")
    if families:
        lines.append(f"- Family: {'; '.join(families)}")
    if parts:
        lines.append(f"- Parts used: {'; '.join(parts)}")
    if diseases:
        lines.append(f"- Uses (Disease used on): {'; '.join(diseases)}")
    if preps:
        lines.append(f"- Preparation and Administration: {'; '.join(preps)}")
    if dosages:
        lines.append(f"- Quality of Dosage: {'; '.join(dosages)}")
    if freqs:
        lines.append(f"- Administration Frequency: {'; '.join(freqs)}")
    if sides:
        lines.append(f"- Experienced adverse or side effects: {'; '.join(sides)}")

    sources = set()
    for row in records:
        lit = _clean_field(row.get("Literature taken from", ""))
        page = _clean_field(row.get("Page where the data can be found on the literature", ""))
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

    detailed_block = "\n".join(lines)
    return intro_paragraph + "\n\n" + detailed_block


def build_prep_method_answer(plant_name: str, condition_phrases: list[str]) -> str:
    """
    Answer questions like:
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

    # 1. Load all rows for this plant
    records = get_raw_records_for_name(plant_name)
    if not records:
        return FALLBACK_MSG

    # Small cleaner used below
    def _clean(val):
        s = str(val)
        if s.lower() == "nan":
            return ""
        return re.sub(r"\s+", " ", s).strip()

    # Helper to collect unique, non-empty values for a column
    def _collect_unique(rows: list[dict], key: str) -> list[str]:
        seen = set()
        out: list[str] = []
        for r in rows:
            v = _clean(r.get(key, ""))
            if v and v not in seen:
                seen.add(v)
                out.append(v)
        return out

    # Build a nice plant label like "Lagundi (Vitex negundo L.)"
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

    # ------------------------------------------------------------
    # Helper: general prep info for this plant (no condition filter)
    # ------------------------------------------------------------
    def _build_general_prep_block() -> str:
        parts   = _collect_unique(records, "Parts used")
        preps   = _collect_unique(records, "Preparation and Administration")
        dosages = _collect_unique(records, "Quality of Dosage")
        freqs   = _collect_unique(records, "Administration Frequency")

        lines: list[str] = []
        lines.append(
            f"Here is the general preparation information recorded for {plant_label}:"
        )

        if parts:
            lines.append(f"- Parts used: {'; '.join(parts)}")
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

    # ------------------------------------------------------------
    # Condition-aware branch: "prep method for lagundi for X"
    # ------------------------------------------------------------

    # Clean condition labels like "for bughat" → "bughat"
    cleaned_conditions: list[str] = []
    for raw in condition_phrases:
        s = clean_condition_label(str(raw))
        if s:
            cleaned_conditions.append(s)

    # Strip plant-name words from conditions, e.g.
    # "peperomia pellucida for skin burns" → "skin burns"
    trimmed_conditions: list[str] = []
    for cond in cleaned_conditions:
        trimmed = strip_plant_name_from_condition(plant_name, cond)
        if trimmed:
            trimmed_conditions.append(trimmed)
    cleaned_conditions = trimmed_conditions

    if not cleaned_conditions:
        # Condition text was too messy → fallback to general prep
        return _build_general_prep_block()

    main_condition = cleaned_conditions[0]

    # Helper: does this row's "Disease used on" mention a condition?
    def _row_mentions_condition(row: dict, cond: str) -> bool:
        disease_text = str(row.get("Disease used on", "")).lower()
        cond = cond.lower().strip()
        if not cond:
            return False

        # direct substring
        if cond in disease_text:
            return True

        # token-based backup (ignore generic words)
        tokens = [
            t for t in re.split(r"[^a-z]+", cond)
            if t and t not in DISEASE_STOPWORDS
        ]
        return any(tok in disease_text for tok in tokens)

    # Filter rows for this plant where the disease mentions ANY of the conditions
    condition_rows: list[dict] = []
    for row in records:
        for cond in cleaned_conditions:
            if _row_mentions_condition(row, cond):
                condition_rows.append(row)
                break

    # ---------- Case 1: condition NOT mentioned for this plant ----------
    if not condition_rows:
        first_line = (
            f"The dataset does not mention **{main_condition}** "
            f"as an indication for {plant_label}."
        )
        general_block = _build_general_prep_block()

        # Highlight condition words (e.g. bughat, cough)
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

    # ---------- Case 2: condition IS mentioned for this plant ----------
    parts   = _collect_unique(condition_rows, "Parts used")
    preps   = _collect_unique(condition_rows, "Preparation and Administration")
    dosages = _collect_unique(condition_rows, "Quality of Dosage")
    freqs   = _collect_unique(condition_rows, "Administration Frequency")

    lines: list[str] = []
    lines.append(
        f"Preparation information for {plant_label} "
        f"when used for **{main_condition}** (according to the dataset):"
    )

    if parts:
        lines.append(f"- Parts used (for **{main_condition}**): {'; '.join(parts)}")
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

    # Highlight condition terms in the final text
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



def condition_matches_row(row: dict, cond: str) -> bool:
    """
    Helper used only by build_prep_method_answer:
    check whether 'Disease used on' text mentions the given condition.
    """
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


# ------------------------------
# 9. MAIN CHAT LOOP
# ------------------------------

def main() -> None:
    """
    Entry point for the ethnomedicinal plant chatbot.

    This version includes:
      - Spellfix (apply_spellfix) integration.
      - A dedicated 'preparation method' branch that uses build_prep_method_answer()
        and NEVER routes through the yes/no condition logic.
    """

    console.print(
        Panel(
            "🌿 Arcana Herbarium – Mindanao Plant Codex",
            style="bold green",
        )
    )

    history: list[str] = []
    last_reviews = None
    last_plant_sci = None

    while True:
        # --------------------------------------------------------
        # 0. Get user input
        # --------------------------------------------------------
        query = console.input(
            "[bold cyan]Inscribe your question into the Arcana Herbarium (press q to close the codex): [/bold cyan]"
        )

        # Allow user to exit.
        if query.strip().lower() == "q":
            console.print(
                "[bold yellow]Closing the codex... may your remedies be effective and your plants well-tended. 🌿[/bold yellow]"
            )
            break

        # --------------------------------------------------------
        # 0.5 Optional spellfix
        # --------------------------------------------------------
        # apply_spellfix is expected to return (corrected_text, did_fix: bool)
        try:
            corrected, did_fix = apply_spellfix(query)
        except NameError:
            # If apply_spellfix is not defined, just skip correction.
            corrected, did_fix = query, False

        if did_fix and corrected != query:
            console.print(
                f"[dim]Auto-corrected your question to: [italic]{corrected}[/italic][/dim]"
            )
            query = corrected

        lower_q = query.lower()

        # 2. "Prep method" questions – always answer from the CSV for a known plant
        plant_for_prep: str | None = None
        prep_conditions: list[str] = []

        if (
            ("prep method" in lower_q or "preparation method" in lower_q or "preparation" in lower_q)
            and ("to treat" in lower_q or " for " in lower_q)
        ):
            # Try to detect the plant directly from the text (scientific name or synonym)
            plant_for_prep = find_plant_in_text(query)

            # If we didn't spot a plant name but we do have a last plant
            # and the user is clearly referring back to it ("it", "this plant"),
            # treat this as a follow-up.
            if not plant_for_prep and last_plant_sci and is_followup_question(query):
                plant_for_prep = last_plant_sci

            if plant_for_prep:
                raw_conditions = extract_condition_phrases_from_query(query)

                # Only keep conditions that are NOT just the plant name itself.
                if is_condition_actually_disease(plant_for_prep, raw_conditions):
                    prep_conditions = raw_conditions
                else:
                    prep_conditions = []

        # If we detected a plant for a "prep method" query, always respond here
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
        # 3. Condition-specific YES/NO questions like:
        #       "Can <plant> treat anemia?"
        #       "Is <plant> used for cough?"
        # --------------------------------------------------------
        plant_for_condition: str | None = None
        condition_phrases: list[str] = []

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
        # 4. Follow-up questions that refer back to the last plant,
        #    e.g. "What about its side effects?"
        # --------------------------------------------------------
        if is_followup_question(query) and last_plant_sci is not None:
            if last_reviews is None:
                last_reviews = retrieve_with_threshold(last_plant_sci)

            reviews = last_reviews
            active_plant = last_plant_sci

        else:
            # --------------------------------------------------------
            # 5. New question – first try to detect a single specific plant.
            # --------------------------------------------------------
            plant_fragment = extract_single_plant_name(query)

            plant_candidate: str | None = None
            if plant_fragment:
                plant_candidate = find_plant_in_text(plant_fragment)

                # Extra safety: exact match to a known plant name
                if plant_candidate is None and plant_fragment in KNOWN_PLANT_NAMES:
                    plant_candidate = plant_fragment

            # 5A. Single-plant CSV answer path
            if plant_candidate is not None:
                records = get_raw_records_for_name(plant_candidate)

                if records:
                    q_low = query.lower()

                    # Disease-focused questions: keep using your usage-summary helper
                    if "what disease" in q_low:
                        # builds: "<plant> can be used to treat the following diseases..."
                        answer = build_usage_summary_answer(plant_candidate)

                    # Plain "what is ..." questions: use the new intro + full details
                    elif (
                        q_low.startswith("what is ")
                        or q_low.startswith("whats ")
                        or q_low.startswith("what's ")
                    ):
                        answer = build_intro_answer_from_records(records)

                    # Everything else: fall back to the simpler summary
                    else:
                        answer = build_answer_from_records(records)

                    print_answer(answer)

                    history.append(f"User: {query}")
                    history.append(f"Assistant: {answer}")

                    # Remember this plant for follow-ups
                    last_plant_sci = str(records[0]["Scientific Name"])
                    last_reviews = retrieve_with_threshold(last_plant_sci)
                    continue  # Done with this question

            # --------------------------------------------------------
            # 5B. General / unclear question – semantic retrieval
            # --------------------------------------------------------
            raw_docs = retrieve_with_threshold(query)

            if not raw_docs:
                print_error(
                    "No matching plant information found in the dataset.\n"
                    "This might mean the plant is not in the dataset, or the spelling is very different."
                )
                continue

            # --------------------------------------------------------
            # 6. Multi-plant vs single-plant handling
            # --------------------------------------------------------
            if is_multi_plant_request(query):

                # 6A. PARTS-based multi-plant search
                if is_parts_based_request(query):
                    part_keywords = extract_parts_keywords_from_query(query)

                    if not part_keywords:
                        part_keywords = ["leaf", "leaves", "lf"]

                    records = search_records_by_parts_keywords(part_keywords)

                    if records:
                        records_by_plant: dict[str, list[dict]] = {}
                        for row in records:
                            sci_name = str(row["Scientific Name"])
                            records_by_plant.setdefault(sci_name, []).append(row)

                        num_plants = len(records_by_plant)
                        unique_parts = sorted({kw.lower() for kw in part_keywords})
                        parts_text = ", ".join(unique_parts)
                        header = (
                            f"**{num_plants} matching plants found for parts:** {parts_text}\n"
                        )

                        all_parts_answers: list[str] = []
                        for sci_name, rows in records_by_plant.items():
                            part_answer = build_answer_from_records(rows)
                            all_parts_answers.append(part_answer)

                        final_answer = header + "\n\n" + "\n\n".join(all_parts_answers)

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

                        final_answer = highlight_keywords(
                            final_answer, list(highlight_terms)
                        )

                        print_answer(final_answer)

                        history.append(f"User: {query}")
                        history.append(f"Assistant: {final_answer}")

                        last_plant_sci = None
                        last_reviews = None
                        continue

                    # If nothing found, fall through to disease-based search.

                # 6B. DISEASE-based multi-plant search
                disease_keywords = extract_disease_keywords_from_query(query)
                if not disease_keywords:
                    disease_keywords = ["cough", "colds"]

                records = search_records_by_disease_keywords(disease_keywords)

                if records:
                    records_by_plant: dict[str, list[dict]] = {}
                    for row in records:
                        sci_name = str(row["Scientific Name"])
                        records_by_plant.setdefault(sci_name, []).append(row)

                    num_plants = len(records_by_plant)
                    unique_conditions = sorted({kw.lower() for kw in disease_keywords})
                    conditions_text = ", ".join(unique_conditions)
                    header = (
                        f"**{num_plants} matching plants found for:** {conditions_text}\n"
                    )

                    all_plants_answers: list[str] = []
                    for sci_name, rows in records_by_plant.items():
                        plant_answer = build_answer_from_records(rows)
                        all_plants_answers.append(plant_answer)

                    final_answer = header + "\n\n" + "\n\n".join(all_plants_answers)

                    # Highlight disease words (cough, jaundice, anemia, etc.)
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

                    final_answer = highlight_keywords(
                        final_answer, list(highlight_terms)
                    )

                    print_answer(final_answer)

                    history.append(f"User: {query}")
                    history.append(f"Assistant: {final_answer}")

                    last_plant_sci = None
                    last_reviews = None
                    continue

                # If multi-plant search also fails, just send the raw docs to LLM.
                focused_docs = raw_docs
                plant_sci = None

            else:
                # 6C. Single/general question – focus docs on one main plant if possible
                focused_docs, plant_sci = focus_on_main_plant(raw_docs)

            reviews = focused_docs
            last_reviews = focused_docs
            last_plant_sci = plant_sci

            active_plant = (
                plant_sci
                if plant_sci
                else ("Multiple plants" if is_multi_plant_request(query) else "Not specified")
            )

        # --------------------------------------------------------
        # 7. Build short chat history for LLM
        # --------------------------------------------------------
        history_text = "\n".join(history[-8:])

        if is_followup_question(query) and last_plant_sci:
            explicit_question = (
                f"For the plant {last_plant_sci} that we discussed earlier: {query}"
            )
        else:
            explicit_question = query

        # --------------------------------------------------------
        # 8. Call LLM with retrieved docs
        # --------------------------------------------------------
        result = chain.invoke(
            {
                "active_plant": active_plant,
                "history": history_text,
                "reviews": reviews,
                "question": explicit_question,
            }
        )

        result = tidy_response(result)

        # --------------------------------------------------------
        # 9. Final safety checks on LLM output
        # --------------------------------------------------------
        if is_fallback_message(result):
            print_error(result)
        else:
            if contains_unknown_scientific_name(result):
                print_error(
                    "The model tried to mention a scientific name that is NOT in the dataset.\n"
                    + FALLBACK_MSG
                )
            else:
                print_answer(result)

        history.append(f"User: {query}")
        history.append(f"Assistant: {result}")


# Standard Python pattern: only run main() when executing this file directly.
if __name__ == "__main__":
    main()
