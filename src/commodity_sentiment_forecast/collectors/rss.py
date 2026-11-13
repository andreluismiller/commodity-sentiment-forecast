from __future__ import annotations
from datetime import timezone
import feedparser, pandas as pd
from dateutil import parser
from commodity_sentiment_forecast.common import load_yaml, normalize_text, sha256_text

def collect(sources_cfg="config/sources.yaml") -> pd.DataFrame:
    cfg=load_yaml(sources_cfg)["rss"]; rows=[]
    if not cfg.get("enabled"): return pd.DataFrame()
    for feed in cfg.get("feeds",[]):
        parsed=feedparser.parse(feed["url"])
        for e in parsed.entries:
            try: dt=parser.parse(e.get("published",e.get("updated"))).astimezone(timezone.utc)
            except Exception: dt=pd.NaT
            rows.append({"url":e.get("link"),"title":normalize_text(e.get("title")),"published_at":dt,"domain":feed.get("domain"),"language":feed.get("language"),"source_country":feed.get("country"),"search_commodity":feed.get("commodity"),"search_query":"rss"})
    df=pd.DataFrame(rows)
    if df.empty:return df
    df["title_hash"]=df["title"].map(sha256_text)
    return df.drop_duplicates("url").drop_duplicates("title_hash").reset_index(drop=True)
