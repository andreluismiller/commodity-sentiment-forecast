import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from commodity_sentiment_forecast.common import normalize_key

def deduplicate(df: pd.DataFrame, threshold=0.92) -> pd.DataFrame:
    if len(df)<2:return df.copy()
    work=df.copy();work["_date"]=pd.to_datetime(work["published_at"],utc=True).dt.date;keep=[]
    for _,g in work.groupby(["_date","search_commodity"],dropna=False):
        if len(g)==1:keep.extend(g.index);continue
        try:
            matrix=TfidfVectorizer(ngram_range=(1,2),max_features=10000).fit_transform(g["title"].fillna("").map(normalize_key));sim=cosine_similarity(matrix);removed=set()
            for i in range(len(g)):
                if i in removed:continue
                keep.append(g.index[i]);removed.update(j for j in range(i+1,len(g)) if sim[i,j]>=threshold)
        except ValueError:keep.extend(g.index)
    return work.loc[sorted(set(keep))].drop(columns="_date").reset_index(drop=True)
