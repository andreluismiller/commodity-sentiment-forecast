import pandas as pd
from src.common import load_yaml, normalize_key

def classify(text,cfg):
    title,_,body=text.partition("."); title=normalize_key(title);body=normalize_key(body);matches=[]
    for commodity,v in cfg.items():
        t=sum(title.count(normalize_key(k)) for k in v["keywords"]);b=sum(body.count(normalize_key(k)) for k in v["keywords"]);score=3*t+b
        if score:matches.append({"commodity":commodity,"keyword_score":score,"relevance_score":min(1.0,score/8)})
    return sorted(matches,key=lambda x:x["keyword_score"],reverse=True)

def add_relevance(df, keywords_cfg="config/keywords.yaml"):
    cfg=load_yaml(keywords_cfg)["commodities"];work=df.copy();work["matches"]=work["analysis_text"].map(lambda x:classify(x,cfg))
    work["matches"]=work.apply(lambda r:r["matches"] or [{"commodity":r["search_commodity"],"keyword_score":1,"relevance_score":0.2}],axis=1)
    work=work.explode("matches",ignore_index=True);details=pd.json_normalize(work.pop("matches"));return pd.concat([work,details],axis=1)
