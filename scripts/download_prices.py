# scripts/download_prices.py
import yfinance as yf, pandas as pd

tickers = {"gold": "GC=F", "silver": "SI=F", "oil": "CL=F"}
frames = []
for name, t in tickers.items():
    d = yf.download(t, start="2020-01-01", end="2026-11-13", auto_adjust=True, progress=False)
    d = d["Close"].squeeze().rename("price").reset_index()
    d.columns = ["date", "price"]
    d["commodity"] = name
    frames.append(d[["date", "commodity", "price"]])

pd.concat(frames).to_csv("data/raw/commodity_prices.csv", index=False)