# vector.py
# -------------------------------------------------------------------
# Responsibilities:
#   1. Load the ethnomedicinal plant CSV dataset into a pandas DataFrame.
#   2. Provide clean, deterministic lookups on the CSV (no AI, no guessing).
#   3. Build and query a Chroma vector database for semantic search.
#
# This module is used by main.py for:
#   - Getting all known plant names (local + scientific).
#   - Fetching all rows for a specific plant.
#   - Searching by disease/condition keywords.
#   - Supplying semantic-retrieval documents to the LLM.
#   - Providing a dataset vocabulary for scope checking.
# -------------------------------------------------------------------

from __future__ import annotations

import os
import re
from typing import List, Dict, Set

import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ===============================================================
# 0. CONFIGURATION
# ===============================================================

# Absolute path to your ethnomedicinal plant dataset
CSV_PATH: str = r"C:/Desktop/ai_chatbot/datasets/Mindanao_Ethnomedicinal_Plant_Dataset.csv"

# Directory on disk where Chroma will store the vector index
CHROMA_DIR: str = "./chroma_mindanao_ethno_db"

# Name for the Chroma collection (just an identifier)
CHROMA_COLLECTION_NAME: str = "Mindanao_Ethnomedicinal_2025"

# Name of the Ollama embedding model to use
EMBED_MODEL_NAME: str = "mxbai-embed-large"


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
    # Keep letters and spaces only
    s = re.sub(r"[^a-zA-Z\s]", " ", str(s))
    # Collapse repeated spaces and lowercase
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

# Read CSV into a pandas DataFrame
df: pd.DataFrame = pd.read_csv(CSV_PATH)

# Strip trailing spaces from headers like "Family " -> "Family"
df.columns = [c.strip() for c in df.columns]

# Column name constants (after stripping spaces)
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

# This set contains *all* distinct local and scientific names
# (as they appear in the CSV, but lowercased).
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
    that actually appear in the dataset.
    """
    return KNOWN_PLANT_NAMES


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

def get_raw_records_for_name(name: str) -> list[dict]:
    """
    Return ALL CSV rows that belong to the same plant,
    treating different author formats (L., Linn., etc.) as the same
    if they share the same Genus + species.
    """
    target = _normalize_sci_name_for_match(name)

    mask = df["Scientific Name"].apply(
        lambda x: _normalize_sci_name_for_match(x) == target
    )

    rows = df[mask]
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

    # Lowercase keywords once
    lowered = [kw.lower() for kw in keywords]

    # Lowercase disease column once
    series = df[COL_DISEASE].astype(str).str.lower()

    def matches(text: str) -> bool:
        # True if ANY keyword appears as substring in text
        return any(kw in text for kw in lowered)

    mask = series.apply(matches)

    records: List[Dict] = [row.to_dict() for _, row in df[mask].iterrows()]
    return records

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
    """
    phrases: Set[str] = set()
    series = df[COL_DISEASE].astype(str).str.lower()

    for text in series:
        # Split by semicolon, comma, or slash
        parts = re.split(r"[;,/]", text)
        for part in parts:
            p = part.strip()
            if p:
                phrases.add(p)

    return sorted(phrases)


# ===============================================================
# 5. EMBEDDINGS + CHROMA VECTOR STORE
# ===============================================================

# Create an embedding model instance backed by Ollama
embeddings = OllamaEmbeddings(model=EMBED_MODEL_NAME)

# Decide if we need to build the Chroma index or just reuse it
add_docs: bool = not os.path.exists(CHROMA_DIR)

if add_docs:
    # We will build a document list from every CSV row
    docs: List[Document] = []
    ids: List[str] = []

    for i, row in df.iterrows():
        # Combine the relevant CSV fields into a single text blob.
        # This is what the embedding model will actually "see".
        page_content = (
            f"Local Name: {row[COL_LOCAL]}\n"
            f"Scientific Name: {row[COL_SCI]}\n"
            f"Disease used on: {row[COL_DISEASE]}\n"
            f"Preparation and Administration: {row[COL_PREP]}\n"
            f"Quality of Dosage: {row[COL_QTY]}\n"
            f"Administration Frequency: {row[COL_FREQ]}\n"
            f"Family: {row[COL_FAMILY]}\n"
            f"Parts used: {row[COL_PARTS]}\n"
            f"Experienced adverse or side effects: {row[COL_SIDE]}\n"
            f"Literature taken from: {row[COL_LIT]}\n"
            f"Page where the data can be found on the literature: {row[COL_PAGE]}\n"
        )

        # Each row becomes one Document with metadata for later use
        doc = Document(
            page_content=page_content,
            metadata={
                "Local Name": row[COL_LOCAL],
                "Scientific Name": row[COL_SCI],
                "Family": row[COL_FAMILY],
                "Parts used": row[COL_PARTS],
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

    Parameters:
        query           – the natural-language query string
        k               – how many top documents to request from Chroma
        score_threshold – minimal similarity score to keep a document

    Returns:
        A list of Document objects representing the most relevant
        dataset rows, filtered by score_threshold.
    """
    results = vector_store.similarity_search_with_score(query=query, k=k)

    # NOTE:
    # Depending on how the underlying distance is defined, "score"
    # might be a similarity (higher = better) or a distance (lower = better).
    # In langchain_chroma's default setup with embeddings, higher scores
    # typically mean "more similar", which is what we assume here.
    filtered_docs: List[Document] = [
        doc for doc, score in results if score >= score_threshold
    ]
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

        # Drop NaN, cast to string
        series = df[col].dropna().astype(str)

        for text in series:
            # Split on non-word characters, lowercase tokens
            for tok in re.split(r"[^\w]+", text.lower()):
                tok = tok.strip()
                # Ignore very short tokens (like "of", "to", etc.)
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
