"""Naive baseline: Random Forest on ECFP4, trained on public training sets,
tested on the SAME external test sets used for the four tools."""
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
from scipy.stats import spearmanr

norm=rdMolStandardize.Normalizer(); lfc=rdMolStandardize.LargestFragmentChooser(); unch=rdMolStandardize.Uncharger()
def std(smi):
    m=Chem.MolFromSmiles(str(smi))
    return unch.uncharge(norm.normalize(lfc.choose(m))) if m else None
def fp(m): return np.array(AllChem.GetMorganFingerprintAsBitVect(m,2,2048))
def featurize(smiles_list):
    X=[]; keep=[]
    for i,s in enumerate(smiles_list):
        m=std(s)
        if m is not None: X.append(fp(m)); keep.append(i)
    return np.array(X), keep
def ik14(s):
    m=std(s); return Chem.MolToInchiKey(m).split('-')[0] if m else None

O='/mnt/user-data/outputs/'
results=[]

def run(name, train_smiles, train_y, test_file, test_smiles_col, kind, exclude_test=True):
    test=pd.read_csv(test_file)
    Xtr,keep=featurize(train_smiles); ytr=np.asarray(train_y)[keep]
    # remove any training compound that appears in the test set (leakage guard)
    test_keys=set(test[test_smiles_col].map(ik14).dropna())
    tr_keys=[ik14(s) for s in np.asarray(train_smiles)[keep]]
    mask=np.array([k not in test_keys for k in tr_keys])
    Xtr,ytr=Xtr[mask],ytr[mask]
    Xte,keep_te=featurize(test[test_smiles_col])
    tt=test.iloc[keep_te].reset_index(drop=True)
    if kind=='clf':
        mdl=RandomForestClassifier(n_estimators=500,random_state=42,n_jobs=-1,class_weight='balanced')
        mdl.fit(Xtr,ytr.astype(int)); p=mdl.predict_proba(Xte)[:,1]
        y=tt['y_true'].values.astype(int)
        r={'endpoint':name,'n_train':len(ytr),'n_test':len(y),
           'AUROC':round(roc_auc_score(y,p),3),'AUPRC':round(average_precision_score(y,p),3),
           'MCC':round(matthews_corrcoef(y,(p>=0.5).astype(int)),3)}
        # per novelty bin
        for b in ['Near','Intermediate','Novel']:
            idx=tt['novelty_bin']==b
            if idx.sum()>=8 and len(np.unique(y[idx.values]))==2:
                r[f'AUROC_{b}']=round(roc_auc_score(y[idx.values],p[idx.values]),3)
        tt['baseline_RF']=np.round(p,3)
    else:
        mdl=RandomForestRegressor(n_estimators=500,random_state=42,n_jobs=-1)
        mdl.fit(Xtr,ytr); p=mdl.predict(Xte)
        y=tt['y_true'].values.astype(float)
        r={'endpoint':name,'n_train':len(ytr),'n_test':len(y),
           'RMSE':round(float(np.sqrt(np.mean((y-p)**2))),3),
           'Spearman':round(float(spearmanr(y,p).correlation),3)}
        for b in ['Near','Intermediate','Novel']:
            idx=(tt['novelty_bin']==b).values
            if idx.sum()>=8: r[f'RMSE_{b}']=round(float(np.sqrt(np.mean((y[idx]-p[idx])**2))),3)
        tt['baseline_RF']=np.round(p,3)
    results.append(r)
    tt[['name','baseline_RF']].to_csv(f'/home/claude/baseline_{name}.csv',index=False)
    print(r)

# --- Solubility: Delaney/ESOL -> SC-2 ---
d=pd.read_csv('sol_train.csv')
solcol=[c for c in d.columns if 'measured log solubility' in c.lower()][0]
run('Solubility', d['smiles'], d[solcol], O+'sc2_solubility_ground_truth.csv','smiles','reg')
# --- hERG: CardioTox train -> external 84 ---
h=pd.read_csv('CardioTox-master/data/train_validation_cardio_tox_data.csv')
run('hERG', h['smiles'], h['ACTIVITY'], O+'herg_ground_truth.csv','smiles','clf')
# --- BBB: BBBP/Martins -> 184 ---
b=pd.read_csv('BBBP.csv')
run('BBB', b['smiles'], b['p_np'], O+'bbb_ground_truth.csv','smiles','clf')
# --- DILI: gold-standard remainder -> 198 ---
g=pd.read_csv('DILI-main/data/DILI_Goldstandard_1111.csv')
run('DILI', g['smiles_r'], g['TOXICITY'], O+'dili_ground_truth.csv','smiles','clf')

pd.DataFrame(results).to_csv(O+'naive_baseline_results.csv',index=False)
print("\nsaved naive_baseline_results.csv")
