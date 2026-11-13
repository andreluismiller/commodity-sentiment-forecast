from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

def prepare(df):
    work=df.sort_values("date").copy();work["date"]=pd.to_datetime(work["date"]);work["month"]=work["date"].dt.month;work["year"]=work["date"].dt.year
    for lag in [1,3,6,12]:work[f"revenue_lag_{lag}"]=work["revenue"].shift(lag)
    work["revenue_ma_3"]=work["revenue"].shift(1).rolling(3).mean();return work

def train(df,test_months=12,random_state=42,output=None):
    work=prepare(df).dropna(subset=["revenue"]);features=[c for c in work.columns if c not in {"date","revenue"} and pd.api.types.is_numeric_dtype(work[c])];tr,te=work.iloc[:-test_months],work.iloc[-test_months:]
    model=Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",RandomForestRegressor(n_estimators=500,min_samples_leaf=2,random_state=random_state,n_jobs=-1))]);model.fit(tr[features],tr["revenue"]);pred=model.predict(te[features]);metrics={"mae":mean_absolute_error(te["revenue"],pred),"rmse":mean_squared_error(te["revenue"],pred)**.5,"train_rows":len(tr),"test_rows":len(te)}
    if output:Path(output).parent.mkdir(parents=True,exist_ok=True);joblib.dump({"model":model,"features":features,"metrics":metrics},output)
    return model,metrics,pd.DataFrame({"date":te["date"],"actual":te["revenue"],"predicted":pred})

def scenarios(point_forecast,residual_std,runs=1000,seed=42):
    rng=np.random.default_rng(seed);sim=rng.normal(point_forecast,residual_std,size=(runs,len(point_forecast)));return pd.DataFrame({"p10":np.quantile(sim,.1,axis=0),"p50":np.quantile(sim,.5,axis=0),"p90":np.quantile(sim,.9,axis=0)})
