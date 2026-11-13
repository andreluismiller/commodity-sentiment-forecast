import re,pandas as pd
from src.common import load_yaml, normalize_key

def classify(text,events):
    text=normalize_key(text);found=[]
    for name,cfg in events.items():
        count=sum(len(re.findall(p,text,re.I)) for p in cfg["patterns"])
        if count:found.append({"event_type":name,"event_count":count,"event_price_impact":cfg["impact"]})
    return max(found,key=lambda x:x["event_count"]) if found else {"event_type":"nao_classificado","event_count":0,"event_price_impact":0}

def add_events(df,keywords_cfg="config/keywords.yaml"):
    events=load_yaml(keywords_cfg)["events"];details=df["analysis_text"].map(lambda x:classify(x,events)).apply(pd.Series)
    return pd.concat([df.reset_index(drop=True),details],axis=1)
