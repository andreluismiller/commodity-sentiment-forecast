from __future__ import annotations
import numpy as np,pandas as pd,torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from commodity_sentiment_forecast.common import load_yaml

class FinBert:
    def __init__(self,model_name,batch_size,max_length):
        self.batch_size=batch_size;self.max_length=max_length;self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer=AutoTokenizer.from_pretrained(model_name);self.model=AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        self.labels={int(k):str(v).lower() for k,v in self.model.config.id2label.items()}
    def predict(self,texts):
        rows=[]
        for start in tqdm(range(0,len(texts),self.batch_size),desc="FinBERT"):
            batch=texts[start:start+self.batch_size];inputs=self.tokenizer(batch,padding=True,truncation=True,max_length=self.max_length,return_tensors="pt");inputs={k:v.to(self.device) for k,v in inputs.items()}
            with torch.no_grad(): probs=torch.softmax(self.model(**inputs).logits,dim=-1).cpu().numpy()
            for p in probs:
                m={self.labels[i]:float(v) for i,v in enumerate(p)};pos=m.get("positive",0);neg=m.get("negative",0);neu=m.get("neutral",0);label=max({"positive":pos,"neutral":neu,"negative":neg},key={"positive":pos,"neutral":neu,"negative":neg}.get)
                rows.append({"sentiment_label":label,"sentiment_positive":pos,"sentiment_neutral":neu,"sentiment_negative":neg,"sentiment_score":pos-neg})
        return pd.DataFrame(rows)

def add_sentiment(df, model_cfg="config/model.yaml"):
    cfg = load_yaml(model_cfg)["nlp"]
    work = df.copy()
    eligible = work["detected_language"].isin(["en", "unknown"])
    num_cols = ["sentiment_positive", "sentiment_neutral", "sentiment_negative", "sentiment_score"]

    work["sentiment_label"] = pd.Series([None] * len(work), index=work.index, dtype="object")
    for c in num_cols:
        work[c] = np.nan

    if eligible.any():
        res = FinBert(cfg["finbert_model"], cfg["batch_size"], cfg["max_length"]).predict(
            work.loc[eligible, "analysis_text"].fillna("").tolist()
        )
        work.loc[eligible, "sentiment_label"] = res["sentiment_label"].to_numpy()
        work.loc[eligible, num_cols] = res[num_cols].to_numpy(dtype=float)
    return work
