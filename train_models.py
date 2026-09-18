"""Reproduce IN6227 Variant 1. Usage: python train_models.py --data dataset.zip --out results"""
import argparse, json, zipfile, time
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--out',default='results'); a=ap.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.data) as z:
        train=pd.read_csv(z.open('dataset/train.csv')); test=pd.read_csv(z.open('dataset/test.csv'))
    audit={}
    for name,d in [('train',train),('test',test)]:
        audit[name]={'rows':len(d),'missing':{k:int(v) for k,v in d.isna().sum().items()},'duplicates':int(d.duplicated().sum()),'labels':d.label.value_counts().to_dict()}
    train=train.dropna(subset=['label']).copy(); test=test.dropna(subset=['label']).copy()
    X=train.drop(columns='label'); y=train.label.eq('yes').astype(int)
    Xt=test.drop(columns='label'); yt=test.label.eq('yes').astype(int)
    num=X.select_dtypes(include='number').columns.tolist(); cat=[c for c in X if c not in num]
    audit['numeric_columns']=num; audit['categorical_columns']=cat
    audit['cross_split_duplicate_features']=int(pd.util.hash_pandas_object(Xt,index=False).isin(pd.util.hash_pandas_object(X,index=False)).sum())
    audit['numeric_summary']=X[num].describe().to_dict()
    q1=X[num].quantile(.25); q3=X[num].quantile(.75); iqr=q3-q1
    audit['iqr_outlier_counts']=((X[num]<q1-1.5*iqr)|(X[num]>q3+1.5*iqr)).sum().to_dict()
    audit['unseen_test_categories']={c:sorted(set(Xt[c].dropna())-set(X[c].dropna())) for c in cat}
    def prep(scale):
        steps=[('impute',SimpleImputer(strategy='median'))]
        if scale: steps.append(('scale',StandardScaler()))
        return ColumnTransformer([('numeric',Pipeline(steps),num),('categorical',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore'))]),cat)])
    candidates=[('Logistic regression',LogisticRegression(max_iter=2000,solver='lbfgs',random_state=42),{'model__C':[.1,1,10]},True),('Random forest',RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=2),{'model__max_depth':[12,None],'model__min_samples_leaf':[2,10]},False)]
    results={}
    def metrics(p,s):
        return {'accuracy':accuracy_score(yt,p),'balanced_accuracy':balanced_accuracy_score(yt,p),'precision':precision_score(yt,p,zero_division=0),'recall':recall_score(yt,p),'f1':f1_score(yt,p),'roc_auc':roc_auc_score(yt,s),'average_precision':average_precision_score(yt,s),'confusion_matrix':confusion_matrix(yt,p).tolist()}
    results['Majority baseline']=metrics(np.zeros(len(yt),dtype=int),np.zeros(len(yt)))
    for name,model,grid,scale in candidates:
        start=time.time()
        search=GridSearchCV(Pipeline([('preprocess',prep(scale)),('model',model)]),grid,scoring='average_precision',cv=StratifiedKFold(3,shuffle=True,random_state=42),n_jobs=1,refit=True)
        search.fit(X,y)
        s=search.predict_proba(Xt)[:,1]; p=(s>=.5).astype(int)
        result=metrics(p,s); result.update(best_params=search.best_params_,cv_average_precision=search.best_score_,cv_std=float(search.cv_results_['std_test_score'][search.best_index_]),seconds=time.time()-start)
        results[name]=result
        pd.DataFrame(search.cv_results_).to_csv(out/(name.replace(' ','_')+'_cv.csv'),index=False)
        pd.DataFrame({'actual':yt.to_numpy(),'probability_yes':s,'prediction':p}).to_csv(out/(name.replace(' ','_')+'_predictions.csv'),index=False)
        print(name,json.dumps(result),flush=True)
    payload={'versions':{'sklearn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__},'audit':audit,'results':results}
    (out/'results.json').write_text(json.dumps(payload,indent=2,default=lambda v:int(v)),encoding='utf-8')
if __name__=='__main__': main()
