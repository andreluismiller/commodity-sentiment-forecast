from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor

def train(df,commodity=None,test_fraction=.2,random_state=42,output=None):
    work=df.dropna(subset=["target_return"]).sort_values("date").copy()
    if commodity:work=work[work["commodity"]==commodity]
    excluded={"date","target_return","price"};features=[c for c in work.columns if c not in excluded and (pd.api.types.is_numeric_dtype(work[c]) or c=="commodity")]
    cut=max(1,int(len(work)*(1-test_fraction)));tr,te=work.iloc[:cut],work.iloc[cut:]
    numeric=[c for c in features if c!="commodity"];categorical=[c for c in features if c=="commodity"]
    prep=ColumnTransformer([("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),numeric),("cat",OneHotEncoder(handle_unknown="ignore"),categorical)])
    model=Pipeline([("prep",prep),("model",HistGradientBoostingRegressor(learning_rate=.05,max_iter=300,max_leaf_nodes=20,l2_regularization=1,random_state=random_state))])
    model.fit(tr[features],tr["target_return"]);pred=model.predict(te[features]);metrics={"mae":mean_absolute_error(te["target_return"],pred),"rmse":mean_squared_error(te["target_return"],pred)**.5,"directional_accuracy":float((np.sign(pred)==np.sign(te["target_return"])).mean()),"train_rows":len(tr),"test_rows":len(te)}
    if output:Path(output).parent.mkdir(parents=True,exist_ok=True);joblib.dump({"model":model,"features":features,"metrics":metrics},output)
    return model,metrics,pd.DataFrame({"date":te["date"],"actual":te["target_return"],"predicted":pred})
