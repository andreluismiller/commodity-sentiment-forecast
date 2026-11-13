import numpy as np,pandas as pd
from src.common import load_yaml,read_table

def add_article_features(df,sources_cfg="config/sources.yaml"):
    cfg=load_yaml(sources_cfg)["source_weights"];default=cfg.get("default",0.65)
    def weight(domain):return next((v for k,v in cfg.items() if k!="default" and (str(domain)==k or str(domain).endswith("."+k))),default)
    work=df.copy();work["source_weight"]=work["domain"].fillna("").map(weight)
    work["expected_price_impact"]=work.apply(lambda r:np.clip(.75*r["event_price_impact"]+.25*(0 if pd.isna(r["sentiment_score"]) else r["sentiment_score"]),-1,1) if r["event_price_impact"] else (0 if pd.isna(r["sentiment_score"]) else r["sentiment_score"]),axis=1)
    work["article_weight"]=work["relevance_score"]*work["source_weight"];return work

def _wavg(v,w):
    ok=v.notna()&w.notna();return np.average(v[ok],weights=w[ok]) if ok.any() and w[ok].sum()>0 else (v[ok].mean() if ok.any() else np.nan)

def aggregate_daily(df,model_cfg="config/model.yaml"):
    cfg=load_yaml(model_cfg)["features"];work=df.copy();work["date"]=pd.to_datetime(work["published_at"],utc=True).dt.tz_localize(None).dt.floor("D");rows=[]
    for (date,commodity),g in work.groupby(["date","commodity"]):
        w=g["article_weight"];rows.append({"date":date,"commodity":commodity,"news_count":g["url"].nunique(),"source_count":g["domain"].nunique(),"positive_share":(g["sentiment_score"]>.15).mean(),"negative_share":(g["sentiment_score"]<-.15).mean(),"sentiment_mean":g["sentiment_score"].mean(),"sentiment_weighted":_wavg(g["sentiment_score"],w),"price_impact_weighted":_wavg(g["expected_price_impact"],w),"supply_event_count":g["event_type"].isin(["interrupcao_oferta","aumento_oferta"]).sum(),"demand_event_count":g["event_type"].isin(["aumento_demanda","queda_demanda"]).sum()})
    daily=pd.DataFrame(rows).sort_values(["commodity","date"]);grp=daily.groupby("commodity",group_keys=False)
    for n in cfg["rolling_windows"]:
        for col in ["sentiment_weighted","price_impact_weighted","news_count"]:
            daily[f"{col}_ma_{n}d"]=grp[col].transform(lambda s:s.shift(1).rolling(n,min_periods=1).mean())
    for n in cfg["lags"]:
        for col in ["sentiment_weighted","price_impact_weighted"]:
            daily[f"{col}_lag_{n}d"]=grp[col].shift(n)
    return daily.reset_index(drop=True)

def build_price_dataset(prices_path,daily_path,horizon=5):
    prices=read_table(prices_path);news=read_table(daily_path);prices["date"]=pd.to_datetime(prices["date"]);news["date"]=pd.to_datetime(news["date"])
    prices=prices.sort_values(["commodity","date"]);g=prices.groupby("commodity");prices["return_1d"]=g["price"].pct_change();prices["volatility_20d"]=g["return_1d"].transform(lambda s:s.rolling(20).std());prices["target_return"]=g["price"].shift(-horizon)/prices["price"]-1
    out=prices.merge(news,on=["date","commodity"],how="left");out["news_count"]=out["news_count"].fillna(0);return out
