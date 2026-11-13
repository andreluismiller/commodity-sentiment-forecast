import argparse,json
from pathlib import Path
from commodity_sentiment_forecast.common import load_yaml,read_table,resolve,setup_logging
from commodity_sentiment_forecast.pipeline import collect_news,process_news,prepare_price_data
from commodity_sentiment_forecast.models.price_model import train as train_price
from commodity_sentiment_forecast.models.revenue_model import train as train_revenue

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest="cmd",required=True)
    for name in ["collect","all"]:
        q=sub.add_parser(name);q.add_argument("--start",required=True);q.add_argument("--end",required=True)
    sub.add_parser("process");q=sub.add_parser("train-price");q.add_argument("--commodity");sub.add_parser("train-revenue")
    a=p.parse_args();setup_logging();cfg=load_yaml("config/model.yaml");paths=cfg["paths"]
    if a.cmd in {"collect","all"}:collect_news(a.start,a.end)
    if a.cmd in {"process","all"}:process_news()
    if a.cmd in {"train-price","all"}:
        df=prepare_price_data();name=getattr(a,"commodity",None) or "all";_,m,_=train_price(df,getattr(a,"commodity",None),cfg["price_model"]["test_fraction"],cfg["price_model"]["random_state"],resolve(paths["models"])/f"price_{name}.joblib");print(json.dumps(m,indent=2))
    if a.cmd in {"train-revenue","all"}:
        df=read_table(paths["revenue"]);_,m,_=train_revenue(df,cfg["revenue_model"]["test_months"],cfg["revenue_model"]["random_state"],resolve(paths["models"])/"revenue.joblib");print(json.dumps(m,indent=2))
if __name__=="__main__":main()
