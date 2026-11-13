from __future__ import annotations
import hashlib, logging, re, unicodedata
from pathlib import Path
from typing import Any
import pandas as pd
import yaml
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]

def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p

def load_yaml(path: str | Path) -> dict:
    with open(resolve(path), encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_parent(path: str | Path) -> Path:
    p = resolve(path); p.parent.mkdir(parents=True, exist_ok=True); return p

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper()), format="%(asctime)s | %(levelname)s | %(message)s")

def normalize_text(value: Any) -> str:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)): return ""
    text = BeautifulSoup(str(value), "html.parser").get_text(" ")
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()

def normalize_key(value: Any) -> str:
    text = unicodedata.normalize("NFKD", normalize_text(value).lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text)).strip()

def sha256_text(value: Any) -> str:
    return hashlib.sha256(normalize_key(value).encode("utf-8")).hexdigest()

def read_table(path: str | Path) -> pd.DataFrame:
    p = resolve(path)
    if p.suffix.lower() == ".csv": return pd.read_csv(p)
    if p.suffix.lower() in {".xlsx", ".xls"}: return pd.read_excel(p, engine="openpyxl")
    if p.suffix.lower() == ".parquet": return pd.read_parquet(p)
    raise ValueError(f"Formato não suportado: {p.suffix}")
