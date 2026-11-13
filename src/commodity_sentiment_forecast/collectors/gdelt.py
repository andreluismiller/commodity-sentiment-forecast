from __future__ import annotations
import logging, time
from datetime import datetime, timedelta, timezone
from typing import Any
import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm.auto import tqdm
from src.common import load_yaml, normalize_text, sha256_text

log = logging.getLogger(__name__)

class GdeltClient:
    def __init__(self, sources_cfg="config/sources.yaml"):
        cfg = load_yaml(sources_cfg); self.cfg = cfg["gdelt"]
        self.url = "https://api.gdeltproject.org/api/v2/doc/doc"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": cfg["articles"]["user_agent"]})

    @retry(retry=retry_if_exception_type(requests.RequestException), stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60), reraise=True)
    def search(self, query: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
        params = {"query": query, "mode": "ArtList", "format": "json", "maxrecords": self.cfg["max_records"], "sort": "DateDesc", "startdatetime": start.strftime("%Y%m%d%H%M%S"), "enddatetime": end.strftime("%Y%m%d%H%M%S")}
        r = self.session.get(self.url, params=params, timeout=self.cfg["request_timeout"]); r.raise_for_status()
        return r.json().get("articles", [])

def _periods(start: datetime, end: datetime, hours: int):
    while start < end:
        nxt = min(start + timedelta(hours=hours), end); yield start, nxt; start = nxt

def collect(start_date: str, end_date: str, keywords_cfg="config/keywords.yaml", sources_cfg="config/sources.yaml") -> pd.DataFrame:
    kws = load_yaml(keywords_cfg)["commodities"]; src = load_yaml(sources_cfg)["gdelt"]
    start = pd.Timestamp(start_date, tz="UTC").to_pydatetime(); end = pd.Timestamp(end_date, tz="UTC").to_pydatetime()
    client = GdeltClient(sources_cfg); rows=[]
    tasks=[(c,q,a,b) for c,v in kws.items() for q in v["queries"] for a,b in _periods(start,end,src["interval_hours"])]
    for commodity, query, a, b in tqdm(tasks, desc="GDELT"):
        try:
            for item in client.search(query,a,b):
                item.update(search_commodity=commodity, search_query=query, search_start=a, search_end=b); rows.append(item)
        except Exception as exc: log.warning("Falha em %s %s-%s: %s", query,a,b,exc)
        time.sleep(src["delay_seconds"])
    if not rows: return pd.DataFrame()
    df=pd.DataFrame(rows).rename(columns={"seendate":"published_at","sourcecountry":"source_country","socialimage":"social_image"})
    for c in ["url","title","published_at","domain","language","source_country","search_commodity","search_query"]:
        if c not in df: df[c]=None
    df["title"]=df["title"].map(normalize_text); df["published_at"]=pd.to_datetime(df["published_at"],utc=True,errors="coerce")
    df["title_hash"]=df["title"].map(sha256_text)
    return df.sort_values("published_at").drop_duplicates("url").drop_duplicates("title_hash").reset_index(drop=True)
