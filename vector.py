# vector.py
# -------------------------------------------------------------------
# Responsibilities:
#   1. Load the ethnomedicinal plant CSV dataset into a pandas DataFrame.
#   2. Provide clean, deterministic lookups on the CSV (no AI, no guessing).
#   3. Build and query a Chroma vector database for semantic search.
#
# Used by main.py for:
#   - Getting all known plant names (local + scientific).
#   - Fetching all rows for a specific plant.
#   - Searching by disease/condition keywords.
#   - Searching by parts used.
#   - Supplying semantic-retrieval documents to the LLM.
#   - Providing a dataset vocabulary for scope checking.
# -------------------------------------------------------------------

from __future__ import annotations

import os
import re
from typing import List, Dict, Set

import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


# ===============================================================
# 0. CONFIGURATION
# ===============================================================

# You can optionally override these via environment variables:
#   ETHNO_DATASET_CSV_PATH
#   ETHNO_CHROMA_DIR
#   ETHNO_CHROMA_COLLECTION
#
# If not set, these defaults are used.

CSV_PATH: str = os.getenv(
    "ETHNO_DATASET_CSV_PATH",
    r"C:/Desktop/improved_ethno_ai_chatbot/datasets/Mindanao_Ethnomedicinal_Plant_Dataset.csv",
)

CHROMA_DIR: str = os.getenv(
    "ETHNO_CHROMA_DIR",
    "./chroma_mindanao_ethno_db",
)

CHROMA_COLLECTION_NAME: str = os.getenv(
    "ETHNO_CHROMA_COLLECTION",
    "Mindanao_Ethnomedicinal_2025",
)

# Sentence-transformers model (multilingual)
EMBED_MODEL_NAME: str = "paraphrase-multilingual-MiniLM-L12-v2"


# ===============================================================
# 1. SMALL TEXT HELPERS
# ===============================================================

def normalize_name(s: str) -> str:
    """
    Normalize a plant name for matching.

    - Lowercase
    - Strip leading/trailing spaces
    - Remove spaces and hyphens completely

    Examples:
        "Tawa tawa"   -> "tawatawa"
        "Tawa-tawa"   -> "tawatawa"
        "  tawa tawA" -> "tawatawa"
    """
    return re.sub(r"[\s\-]+", "", str(s).strip().lower())


def sci_genus_species_normalized(s: str) -> str:
    """
    Normalize a scientific name down to just "genus species",
    all lowercase, with a single space.

    Examples:
        "Andrographis paniculata Nees"
            -> "andrographis paniculata"
        "Andrographis paniculata (Burm. f.) Nees"
            -> "andrographis paniculata"
    """
    s = re.sub(r"[^a-zA-Z\s]", " ", str(s))
    s = re.sub(r"\s+", " ", s).strip().lower()
    parts = s.split()
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"
    return s


def sci_genus_species(s: str) -> str:
    """
    Extract "genus species" from a scientific name string,
    returning it in lowercase.

    Examples:
        "Andrographis paniculata Nees"
            -> "andrographis paniculata"
        "Andrographis paniculata (Burm.f.) Wall. Nees"
            -> "andrographis paniculata"
    """
    s = re.sub(r"[^a-zA-Z\s]", " ", str(s)).lower()
    parts = s.split()
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"
    return " ".join(parts)


# ===============================================================
# 2. LOAD CSV INTO DATAFRAME
# ===============================================================

def _load_csv(path: str) -> pd.DataFrame:
    """
    Load the ethnomedicinal CSV with basic validation and
    friendly error messages if something is wrong.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[vector.py] CSV dataset not found at:\n  {path}\n\n"
            f"Please check that the file exists or set ETHNO_DATASET_CSV_PATH."
        )

    df = pd.read_csv(path)

    # Strip trailing spaces from headers like "Family " -> "Family"
    df.columns = [c.strip() for c in df.columns]

    required_cols = [
        "Scientific Name",
        "Family",
        "Local Name",
        "Disease used on",
        "Parts used",
        "Preparation and Administration",
        "Quality of Dosage",
        "Administration Frequency",
        "Experienced adverse or side effects",
        "Literature taken from",
        "Page where the data can be found on the literature",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "[vector.py] CSV is missing required columns:\n"
            f"  {', '.join(missing)}\n\n"
            "Please make sure you are using the correct dataset version."
        )

    return df


# Global DataFrame (loaded once)
df: pd.DataFrame = _load_csv(CSV_PATH)

# Column name constants
COL_SCI: str = "Scientific Name"
COL_FAMILY: str = "Family"
COL_LOCAL: str = "Local Name"
COL_DISEASE: str = "Disease used on"
COL_PARTS: str = "Parts used"
COL_PREP: str = "Preparation and Administration"
COL_QTY: str = "Quality of Dosage"
COL_FREQ: str = "Administration Frequency"
COL_SIDE: str = "Experienced adverse or side effects"
COL_LIT: str = "Literature taken from"
COL_PAGE: str = "Page where the data can be found on the literature"

# -------------------------------------------------------------------
# Precompute helper columns for faster + cleaner lookups
# -------------------------------------------------------------------

# Normalized local name (for matching "tawa-tawa" vs "tawa tawa")
df["_local_norm"] = df[COL_LOCAL].astype(str).apply(normalize_name)

# Normalized scientific name (full string)
df["_sci_norm"] = df[COL_SCI].astype(str).apply(normalize_name)

# Normalized "genus species" form of the scientific name
df["_gs"] = df[COL_SCI].astype(str).apply(sci_genus_species_normalized)


# ===============================================================
# 3. KNOWN PLANT NAMES (for quick existence checks)
# ===============================================================

KNOWN_PLANT_NAMES: Set[str] = set()

for _, row in df.iterrows():
    local = str(row[COL_LOCAL]).strip().lower()
    sci = str(row[COL_SCI]).strip().lower()
    if local:
        KNOWN_PLANT_NAMES.add(local)
    if sci:
        KNOWN_PLANT_NAMES.add(sci)


def get_known_plant_names() -> Set[str]:
    """
    Return the set of all plant names (local + scientific)
    that actually appear in the dataset (lowercased).
    """
    return KNOWN_PLANT_NAMES


def get_name_to_scientific_map() -> Dict[str, str]:
    """
    Build a dictionary mapping many possible name variants to the
    canonical scientific name exactly as it appears in the CSV.

    Keys are all lowercase and include:
      - the full scientific name
      - the 'Genus species' form of the scientific name
      - a compact normalized scientific name (no spaces / hyphens)
      - the local name
      - compact normalized local name
      - simple hyphen/space variants of the local name

    This lets the chatbot recognize queries like 'what is tanglad',
    'tawa tawa', 'tawatawa', etc. for any plant present in the CSV.
    """
    mapping: Dict[str, str] = {}

    for _, row in df.iterrows():
        sci = str(row[COL_SCI]).strip()
        local = str(row[COL_LOCAL]).strip()

        if not sci:
            continue

        canonical = sci  # scientific name exactly as in the CSV
        aliases: Set[str] = set()

        # ---- scientific name variants ----
        sci_lower = sci.lower()
        aliases.add(sci_lower)
        aliases.add(normalize_name(sci))       # no spaces/hyphens
        aliases.add(sci_genus_species(sci))    # genus + species only

        # ---- local name variants ----
        if local:
            lower_local = local.lower().strip()
            if lower_local:
                aliases.add(lower_local)
                aliases.add(normalize_name(local))         # no spaces/hyphens
                aliases.add(lower_local.replace("-", " ")) # "tawa-tawa" -> "tawa tawa"
                aliases.add(lower_local.replace(" ", ""))  # "tawa tawa" -> "tawatawa"

        # register all aliases → canonical sci name
        for alias in aliases:
            key = alias.strip().lower()
            if not key:
                continue
            mapping.setdefault(key, canonical)

    return mapping


# ===============================================================
# 4. ROW-LEVEL LOOKUPS
# ===============================================================

def _normalize_sci_name_for_match(s: str) -> str:
    """
    Normalize a scientific name for grouping:

    - remove newlines
    - collapse spaces
    - take only the first two words (Genus + species)
    - lowercase
    """
    if s is None:
        return ""
    s = str(s).replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    parts = s.split()
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}".lower()
    return s.lower()


def get_raw_records_for_name(name: str) -> List[Dict]:
    """
    Return ALL CSV rows that belong to the same plant.

    Supports:
      - Scientific names (full or 'Genus species')
      - Local names (e.g. 'tanglad', 'lagundi')

    It first tries to match by scientific name.
    If nothing is found, it will automatically try the Local Name column.
    """
    if not name:
        return []

    # 1) Try to match as scientific name (Genus + species)
    target = _normalize_sci_name_for_match(name)
    sci_mask = df[COL_SCI].apply(
        lambda x: _normalize_sci_name_for_match(x) == target
    )
    rows = df[sci_mask]

    # 2) If nothing found, try to match as LOCAL NAME (normalized)
    if rows.empty:
        local_target = normalize_name(name)
        local_mask = df["_local_norm"] == local_target
        rows = df[local_mask]

    return rows.to_dict(orient="records")


def search_records_by_disease_keywords(keywords: List[str]) -> List[Dict]:
    """
    Find ALL rows where the 'Disease used on' column contains
    at least one of the specified keywords (case-insensitive).

    Used for multi-plant questions like:
      "list of plants that can treat jaundice"
    """
    if not keywords:
        return []

    lowered = [kw.lower() for kw in keywords]
    series = df[COL_DISEASE].astype(str).str.lower()

    def matches(text: str) -> bool:
        return any(kw in text for kw in lowered)

    mask = series.apply(matches)
    return [row.to_dict() for _, row in df[mask].iterrows()]


def search_records_by_parts_keywords(keywords: List[str]) -> List[Dict]:
    """
    Find all CSV rows where the 'Parts used' column contains
    at least one of the specified keywords (case-insensitive).

    Used for queries like:
      - "list all plants that use leaves as medicinal parts"
      - "plants that use bark or roots"
    """
    if not keywords:
        return []

    lowered = [kw.lower() for kw in keywords]
    series = df[COL_PARTS].astype(str).str.lower()

    def matches(text: str) -> bool:
        return any(kw in text for kw in lowered)

    mask = series.apply(matches)
    return [row.to_dict() for _, row in df[mask].iterrows()]


def get_all_disease_phrases() -> List[str]:
    """
    Extract all DISTINCT disease phrases from the 'Disease used on' column.

    We split on common separators ; , and /, then trim whitespace and
    deduplicate the resulting pieces.

    Returned phrases are lowercased.
    """
    phrases: Set[str] = set()
    series = df[COL_DISEASE].astype(str).str.lower()

    for text in series:
        parts = re.split(r"[;,/]", text)
        for part in parts:
            p = part.strip()
            if p:
                phrases.add(p)

    return sorted(phrases)


# ===============================================================
# 5. EMBEDDINGS + CHROMA VECTOR STORE
# ===============================================================

class SBERTEmbeddings(Embeddings):
    """
    Wrapper around SentenceTransformer so that Chroma can call
    embed_documents() and embed_query() like it expects.
    """

    def __init__(self, model_name: str):
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embs = self._model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embs.tolist()

    def embed_query(self, text: str) -> List[float]:
        emb = self._model.encode(
            [text],
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]
        return emb.tolist()


# Single embeddings instance used by Chroma
embeddings = SBERTEmbeddings(EMBED_MODEL_NAME)

# Decide if we need to build the Chroma index or just reuse it
add_docs: bool = not os.path.exists(CHROMA_DIR)

if add_docs:
    docs: List[Document] = []
    ids: List[str] = []

    for i, row in df.iterrows():
        page_content = (
            f"Local Name: {row[COL_LOCAL]}\n"
            f"Scientific Name: {row[COL_SCI]}\n"
            f"Family: {row[COL_FAMILY]}\n"
            f"Disease used on: {row[COL_DISEASE]}\n"
            f"Parts used: {row[COL_PARTS]}\n"
            f"Preparation and Administration: {row[COL_PREP]}\n"
            f"Quality of Dosage: {row[COL_QTY]}\n"
            f"Administration Frequency: {row[COL_FREQ]}\n"
            f"Experienced adverse or side effects: {row[COL_SIDE]}\n"
            f"Literature taken from: {row[COL_LIT]}\n"
            f"Page where the data can be found on the literature: {row[COL_PAGE]}\n"
        )

        doc = Document(
            page_content=page_content,
            metadata={
                "Local Name": row[COL_LOCAL],
                "Scientific Name": row[COL_SCI],
                "Family": row[COL_FAMILY],
                "Disease used on": row[COL_DISEASE],
                "Parts used": row[COL_PARTS],
                "Preparation and Administration": row[COL_PREP],
                "Quality of Dosage": row[COL_QTY],
                "Administration Frequency": row[COL_FREQ],
                "Experienced adverse or side effects": row[COL_SIDE],
                "Literature taken from": row[COL_LIT],
                "Page where the data can be found on the literature": row[COL_PAGE],
            },
            id=str(i),
        )

        ids.append(str(i))
        docs.append(doc)

# Create or connect to the Chroma vector store
vector_store = Chroma(
    collection_name=CHROMA_COLLECTION_NAME,
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)

# If this is the first run (no directory yet), add all documents and persist
if add_docs:
    vector_store.add_documents(documents=docs, ids=ids)


def retrieve_with_threshold(
    query: str,
    k: int = 30,
    score_threshold: float = 0.3,
) -> List[Document]:
    """
    Retrieve documents from the vector store based on semantic similarity.

    For Chroma, the score is usually a *distance*:
      lower = more similar, higher = less similar.

    Behavior:
      - Keep docs with score <= score_threshold.
      - If that gives nothing, fall back to returning the top-k docs
        so the chatbot always has something to work with.
    """
    results = vector_store.similarity_search_with_score(query=query, k=k)

    if not results:
        return []

    # score is a distance: LOWER = closer
    filtered_docs: List[Document] = [
        doc for doc, score in results if score <= score_threshold
    ]

    # If threshold was too strict, just return the top-k docs
    if not filtered_docs:
        filtered_docs = [doc for doc, _ in results]

    return filtered_docs


# ===============================================================
# 6. DATASET VOCABULARY (for scope checking in main.py)
# ===============================================================

def _build_dataset_vocabulary() -> Set[str]:
    """
    Build a set of ALL content words that appear in the core
    columns of the dataset.

    Core columns (included):
        - Scientific Name
        - Family
        - Local Name
        - Disease used on
        - Parts used
        - Preparation and Administration
        - Quality of Dosage
        - Administration Frequency
        - Experienced adverse or side effects

    We intentionally IGNORE:
        - Literature taken from
        - Page where the data can be found...

    Words that appear ONLY in literature titles (e.g. "Manobo")
    will not be considered part of the dataset vocabulary,
    which is useful for "out-of-scope" detection in main.py.
    """
    vocab: Set[str] = set()

    core_cols = [
        COL_SCI,
        COL_FAMILY,
        COL_LOCAL,
        COL_DISEASE,
        COL_PARTS,
        COL_PREP,
        COL_QTY,
        COL_FREQ,
        COL_SIDE,
    ]

    for col in core_cols:
        if col not in df.columns:
            continue

        series = df[col].dropna().astype(str)

        for text in series:
            for tok in re.split(r"[^\w]+", text.lower()):
                tok = tok.strip()
                if len(tok) < 3:
                    continue
                vocab.add(tok)

    return vocab


# Precompute once at import time for fast lookups
_DATASET_VOCAB: Set[str] = _build_dataset_vocabulary()


def get_dataset_vocabulary() -> Set[str]:
    """
    Return the precomputed set of tokens that appear somewhere
    in the core dataset columns.
    """
    return _DATASET_VOCAB


if __name__ == "__main__":
    # Tiny self-check you can run with:
    #   python vector.py
    print("[vector.py] Loaded rows:", len(df))
    print("[vector.py] Example known names (first 10):")
    for i, name in enumerate(sorted(KNOWN_PLANT_NAMES)):
        if i >= 10:
            break
        print("  -", name)
