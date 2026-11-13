import pandas as pd
from src.common import load_yaml,ensure_parent,read_table,resolve
from src.collectors.gdelt import collect as collect_gdelt
from src.collectors.rss import collect as collect_rss
from src.preprocessing.cleaner import clean
from src.preprocessing.deduplicator import deduplicate
from src.nlp.relevance import add_relevance
from src.nlp.event_classifier import add_events
from src.nlp.sentiment import add_sentiment
from src.features.build_features import add_article_features,aggregate_daily,build_price_dataset

def collect_news(start,end,model_cfg="config/model.yaml"):
    paths=load_yaml(model_cfg)["paths"];frames=[collect_gdelt(start,end),collect_rss()];df=pd.concat([x for x in frames if not x.empty],ignore_index=True) if any(not x.empty for x in frames) else pd.DataFrame()
    if not df.empty:df=df.drop_duplicates("url");df.to_parquet(ensure_parent(paths["bronze_news"]),index=False)
    return df

def process_news(model_cfg="config/model.yaml"):
    cfg=load_yaml(model_cfg);paths=cfg["paths"];df=pd.read_parquet(resolve(paths["bronze_news"]));df=clean(df);df=deduplicate(df,cfg["nlp"]["duplicate_threshold"]);df=add_relevance(df);df=df[df["relevance_score"]>=.2];df=add_events(df);df=add_sentiment(df);df=add_article_features(df);df.to_parquet(ensure_parent(paths["silver_news"]),index=False);daily=aggregate_daily(df);daily.to_parquet(ensure_parent(paths["gold_daily"]),index=False);return df,daily

def prepare_price_data(model_cfg="config/model.yaml"):
    cfg=load_yaml(model_cfg);p=cfg["paths"];df=build_price_dataset(p["prices"],p["gold_daily"],cfg["price_model"]["target_horizon"]);df.to_parquet(ensure_parent(p["price_dataset"]),index=False);return df
