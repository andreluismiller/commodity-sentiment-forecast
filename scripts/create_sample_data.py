from pathlib import Path
import numpy as np,pandas as pd


rng=np.random.default_rng(42);Path("data/raw").mkdir(parents=True,exist_ok=True)
dates=pd.date_range("2024-01-01",periods=500,freq="B");rows=[]
for commodity,start in {"cobre":8500,"aluminio":2300,"aco_carbono":700,"aco_eletrico_silicio":1800,"oleo_isolante":1100}.items():
    price=start*np.exp(np.cumsum(rng.normal(0,.01,len(dates))));rows.extend({"date":d,"commodity":commodity,"price":p,"usd_brl":5+rng.normal(0,.08)} for d,p in zip(dates,price))
pd.DataFrame(rows).to_csv("data/raw/commodity_prices.csv",index=False)
m=pd.date_range("2020-01-01",periods=72,freq="MS");rev=50_000_000+np.arange(72)*200_000+rng.normal(0,3_000_000,72)
pd.DataFrame({"date":m,"revenue":rev,"volume":rng.integers(60,100,72),"backlog":rng.uniform(80e6,180e6,72),"avg_sale_price":rng.uniform(500e3,900e3,72),"usd_brl":rng.normal(5,.3,72),"copper_price":rng.normal(9000,900,72),"aluminum_price":rng.normal(2400,250,72),"carbon_steel_price":rng.normal(700,80,72),"electrical_steel_price":rng.normal(1800,200,72),"insulating_oil_price":rng.normal(1100,120,72)}).to_csv("data/raw/revenue_monthly.csv",index=False)
print("Dados sintéticos criados apenas para teste técnico.")
