from __future__ import annotations
import time, logging
import pandas as pd
import requests, trafilatura
from langdetect import detect, LangDetectException
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm.auto import tqdm
from src.common import load_yaml, normalize_text, sha256_text
log=logging.getLogger(__name__)

@retry(retry=retry_if_exception_type(requests.RequestException),stop=stop_after_attempt(3),wait=wait_exponential(min=1,max=10),reraise=True)
def _download(url,cfg):
    r=requests.get(url,headers={"User-Agent":cfg["user_agent"]},timeout=cfg["request_timeout"]);r.raise_for_status();return r.text

def extract(url,cfg):
    try:return normalize_text(trafilatura.extract(_download(url,cfg),include_comments=False,include_tables=False,favor_precision=True,deduplicate=True))
    except Exception as exc: log.debug("Extração falhou %s: %s",url,exc);return ""

def language(text):
    try:return detect(text) if len(text)>=30 else "unknown"
    except LangDetectException:return "unknown"

def clean(df: pd.DataFrame,sources_cfg="config/sources.yaml") -> pd.DataFrame:
    if df.empty:return df.copy()
    cfg=load_yaml(sources_cfg)["articles"]; out=df.copy(); bodies=[]
    for u in tqdm(out["url"],desc="Artigos"):
        bodies.append(extract(u,cfg));time.sleep(cfg["delay_seconds"])
    out["body"]=bodies;out["body_length"]=out["body"].str.len()
    out["analysis_text"]=out["title"].fillna("")+". "+out["body"].where(out["body_length"]>=cfg["min_body_length"],"")
    out["detected_language"]=out["analysis_text"].map(language);out["text_hash"]=out["analysis_text"].map(sha256_text)
    return out.drop_duplicates("text_hash").reset_index(drop=True)
