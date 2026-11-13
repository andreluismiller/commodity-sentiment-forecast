import numpy as np,pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error,mean_squared_error

def walk_forward(model,X,y,min_train=100,step=20):
    rows=[]
    for end in range(min_train,len(X),step):
        stop=min(end+step,len(X));m=clone(model);m.fit(X.iloc[:end],y.iloc[:end]);p=m.predict(X.iloc[end:stop])
        rows.append({"train_end":end,"test_end":stop,"mae":mean_absolute_error(y.iloc[end:stop],p),"rmse":mean_squared_error(y.iloc[end:stop],p)**.5,"directional_accuracy":float((np.sign(p)==np.sign(y.iloc[end:stop])).mean())})
    return pd.DataFrame(rows)

def compare(actual,baseline,candidate):
    a=np.asarray(actual);b=np.asarray(baseline);c=np.asarray(candidate);mae_b=np.mean(np.abs(a-b));mae_c=np.mean(np.abs(a-c));return {"baseline_mae":mae_b,"candidate_mae":mae_c,"gain_percent":100*(mae_b-mae_c)/mae_b if mae_b else np.nan}
