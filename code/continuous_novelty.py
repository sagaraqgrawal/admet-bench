"""Continuous similarity-performance analysis: replaces arbitrary 3-bin cuts."""
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

O='/mnt/user-data/outputs/'
df=pd.read_csv(O+'per_compound_predictions.csv')
COL={'SwissADME':'#2E5B9E','pkCSM':'#C0392B','ADMETlab3.0':'#27AE60','ADMET-AI':'#E67E22'}

def rolling_auc(sub,tool,window=60,step=4):
    """Sliding-window AUROC ordered by Tanimoto similarity."""
    d=sub[['nn_tanimoto','ground_truth',tool]].apply(pd.to_numeric,errors='coerce').dropna()
    d=d.sort_values('nn_tanimoto').reset_index(drop=True)
    xs,ys=[],[]
    for start in range(0,len(d)-window+1,step):
        w=d.iloc[start:start+window]
        if w['ground_truth'].nunique()<2: continue
        try:
            a=roc_auc_score(w['ground_truth'],w[tool])
        except Exception: continue
        xs.append(w['nn_tanimoto'].median()); ys.append(a)
    return np.array(xs),np.array(ys)

def rolling_err(sub,tool,window=40,step=3):
    """Sliding-window RMSE ordered by Tanimoto (regression)."""
    d=sub[['nn_tanimoto','ground_truth',tool]].apply(pd.to_numeric,errors='coerce').dropna()
    d=d.sort_values('nn_tanimoto').reset_index(drop=True)
    xs,ys=[],[]
    for start in range(0,len(d)-window+1,step):
        w=d.iloc[start:start+window]
        xs.append(w['nn_tanimoto'].median())
        ys.append(np.sqrt(np.mean((w['ground_truth']-w[tool])**2)))
    return np.array(xs),np.array(ys)

# also: Spearman correlation between per-compound |error| and Tanimoto (regression)
# and point-biserial style trend test for classification via logistic of correctness
from scipy.stats import spearmanr
trend=[]

fig,axes=plt.subplots(2,2,figsize=(13,9))
panels=[('Solubility (logS0)','Solubility — rolling RMSE (lower=better)',True),
        ('hERG (blocker@10uM)','hERG — rolling AUROC',False),
        ('BBB (logBB>-1)','BBB — rolling AUROC',False),
        ('DILI','DILI — rolling AUROC',False)]
for ax,(ep,title,is_reg) in zip(axes.flat,panels):
    sub=df[df['endpoint']==ep]
    tools=['SwissADME','pkCSM','ADMETlab3.0','ADMET-AI'] if is_reg else ['ADMETlab3.0','ADMET-AI','pkCSM']
    for t in tools:
        if t not in sub.columns: continue
        d=sub[['nn_tanimoto','ground_truth',t]].apply(pd.to_numeric,errors='coerce').dropna()
        if len(d)<50: continue
        if is_reg:
            x,y=rolling_err(sub,t)
            # trend: does |error| correlate with similarity?
            err=(d['ground_truth']-d[t]).abs()
            rho,p=spearmanr(d['nn_tanimoto'],err)
            trend.append({'endpoint':ep,'tool':t,'metric':'|error| vs Tc','spearman_rho':round(rho,3),'p_value':round(p,4)})
        else:
            x,y=rolling_auc(sub,t)
            # trend: correctness vs similarity
            lab=(d[t]>=0.5).astype(int) if d[t].max()<=1 else d[t].astype(int)
            correct=(lab==d['ground_truth']).astype(int)
            rho,p=spearmanr(d['nn_tanimoto'],correct)
            trend.append({'endpoint':ep,'tool':t,'metric':'correct vs Tc','spearman_rho':round(rho,3),'p_value':round(p,4)})
        if len(x)>1:
            ax.plot(x,y,'-',color=COL.get(t,'#666'),lw=2.2,label=t,alpha=0.9)
    ax.axvline(0.4,ls=':',c='#999',lw=1.2); ax.axvline(0.6,ls=':',c='#999',lw=1.2)
    ax.text(0.4,ax.get_ylim()[1],' Tc=0.4',fontsize=7,color='#777',va='top')
    if not is_reg: ax.axhline(0.5,ls='--',c='#bbb',lw=1)
    ax.set_title(title,fontsize=11,weight='bold',loc='left')
    ax.set_xlabel('Nearest-neighbour Tanimoto to approved drugs (Tc)')
    ax.set_ylabel('RMSE (log mol/L)' if is_reg else 'AUROC')
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
fig.suptitle('Continuous novelty–performance relationship (sliding window over Tanimoto similarity)',fontsize=13,weight='bold')
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(O+'figure2_continuous_novelty.png',dpi=150,bbox_inches='tight')
td=pd.DataFrame(trend); td.to_csv(O+'novelty_trend_tests.csv',index=False)
print(td.to_string(index=False))
print('\nsaved figure2_continuous_novelty.png + novelty_trend_tests.csv')
