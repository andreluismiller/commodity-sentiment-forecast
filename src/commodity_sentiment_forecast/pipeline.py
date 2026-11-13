import pandas as pd
from commodity_sentiment_forecastcommon import load_yaml,ensure_parent,read_table,resolve
from commodity_sentiment_forecast.collectors.gdelt import collect as collect_gdelt
from commodity_sentiment_forecast.collectors.rss import collect as collect_rss
from commodity_sentiment_forecast.preprocessing.cleaner import clean
from commodity_sentiment_forecast.preprocessing.deduplicator import deduplicate
from commodity_sentiment_forecast.nlp.relevance import add_relevance
from commodity_sentiment_forecast.nlp.event_classifier import add_events
from commodity_sentiment_forecast.nlp.sentiment import add_sentiment
from commodity_sentiment_forecast.features.build_features import add_article_features,aggregate_daily,build_price_dataset

def collect_news(start,end,model_cfg="config/model.yaml"):
    paths=load_yaml(model_cfg)["paths"];frames=[collect_gdelt(start,end),collect_rss()];df=pd.concat([x for x in frames if not x.empty],ignore_index=True) if any(not x.empty for x in frames) else pd.DataFrame()
    if not df.empty:df=df.drop_duplicates("url");df.to_parquet(ensure_parent(paths["bronze_news"]),index=False)
    return df

def process_news(model_cfg="config/model.yaml"):
    cfg=load_yaml(model_cfg);paths=cfg["paths"];df=pd.read_parquet(resolve(paths["bronze_news"]));df=clean(df);df=deduplicate(df,cfg["nlp"]["duplicate_threshold"]);df=add_relevance(df);df=df[df["relevance_score"]>=.2];df=add_events(df);df=add_sentiment(df);df=add_article_features(df);df.to_parquet(ensure_parent(paths["silver_news"]),index=False);daily=aggregate_daily(df);daily.to_parquet(ensure_parent(paths["gold_daily"]),index=False);return df,daily

def prepare_price_data(model_cfg="config/model.yaml"):
    cfg=load_yaml(model_cfg);p=cfg["paths"];df=build_price_dataset(p["prices"],p["gold_daily"],cfg["price_model"]["target_horizon"]);df.to_parquet(ensure_parent(p["price_dataset"]),index=False);return df
