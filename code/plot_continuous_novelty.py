import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
O='/mnt/user-data/outputs/'
pred=pd.read_csv(O+'per_compound_predictions.csv')
COL={'SwissADME':'#2E5B9E','pkCSM':'#C0392B','ADMETlab3.0':'#27AE60','ADMET-AI':'#E67E22','Baseline RF':'#7F8C8D'}

def slide(df,tool,kind,window,step=3):
    d=df[['ground_truth',tool,'nn_tanimoto']].apply(pd.to_numeric,errors='coerce').dropna()
    d=d.sort_values('nn_tanimoto').reset_index(drop=True)
    xs,ys=[],[]
    for s in range(0,len(d)-window+1,step):
        w=d.iloc[s:s+window]
        if kind=='clf':
            if len(np.unique(w['ground_truth']))<2: continue
            ys.append(roc_auc_score(w['ground_truth'],w[tool]))
        else:
            ys.append(float(np.sqrt(np.mean((w['ground_truth']-w[tool])**2))))
        xs.append(w['nn_tanimoto'].median())
    return xs,ys

fig,axes=plt.subplots(2,3,figsize=(15,8.2))
plt.subplots_adjust(hspace=0.42,wspace=0.3)
panels=[('Solubility (logS0)','reg','A. Solubility',['SwissADME','ADMET-AI','ADMETlab3.0','pkCSM'],40),
        ('hERG (blocker@10uM)','clf','B. hERG',['ADMETlab3.0','ADMET-AI'],40),
        ('BBB (logBB>-1)','clf','C. BBB',['ADMET-AI','pkCSM','ADMETlab3.0'],55),
        ('DILI','clf','D. DILI',['ADMETlab3.0','ADMET-AI'],60)]
pos=[(0,0),(0,1),(1,0),(1,1)]
for (ep,kind,title,tools,win),(r,c) in zip(panels,pos):
    ax=axes[r][c]; sub=pred[pred.endpoint==ep]
    for t in tools:
        xs,ys=slide(sub,t,kind,win)
        if xs: ax.plot(xs,ys,'-',color=COL[t],lw=2.1,label=t,alpha=0.92)
    ax.axvline(0.4,ls=':',c='#999',lw=1.3)
    if kind=='clf':
        ax.axhline(0.5,ls='--',c='#ccc',lw=1); ax.set_ylabel('AUROC'); ax.set_ylim(0.35,1.02)
    else:
        ax.set_ylabel('RMSE (lower better)'); ax.invert_yaxis()
    ax.set_xlabel('NN Tanimoto to approved drugs',fontsize=9)
    ax.set_title(title+' — sliding-window performance',fontsize=10,loc='left',weight='bold')
    ax.grid(alpha=0.22); ax.legend(fontsize=7.5,loc='best')

# merge right column into one tall panel
for r in range(2): axes[r][2].remove()
axE=fig.add_subplot(1,3,3)
rows=[('Solubility (RMSE↓)',{'SwissADME':1.033,'ADMET-AI':1.140,'ADMETlab3.0':1.164,'pkCSM':1.438,'Baseline RF':1.701},False),
      ('hERG (AUROC↑)',{'ADMETlab3.0':0.840,'Baseline RF':0.756,'ADMET-AI':0.748},True),
      ('BBB (AUROC↑)',{'ADMET-AI':0.804,'Baseline RF':0.777,'pkCSM':0.724,'ADMETlab3.0':0.677},True),
      ('DILI (AUROC↑)',{'ADMETlab3.0':0.655,'ADMET-AI':0.611,'Baseline RF':0.609},True)]
y=0; ticks=[]; labs=[]
for name,d,higher in rows:
    ticks.append(y); labs.append(f'$\\bf{{{name.split()[0]}}}$'); y+=0.9
    for tool,v in sorted(d.items(),key=lambda x:(-x[1] if higher else x[1])):
        axE.barh(y,v,color=COL.get(tool,'#999'),edgecolor='white',height=0.7,
                 hatch='///' if tool=='Baseline RF' else None)
        axE.text(v+0.015,y,f'{v:.2f}',va='center',fontsize=7.5)
        ticks.append(y); labs.append(tool); y+=1
    y+=0.7
axE.set_yticks(ticks); axE.set_yticklabels(labs,fontsize=8)
axE.invert_yaxis(); axE.grid(axis='x',alpha=0.25); axE.set_xlim(0,1.85)
axE.set_title('E. Tools vs naive ECFP4 Random-Forest baseline\n(hatched grey = baseline)',fontsize=10,loc='left',weight='bold')
axE.set_xlabel('metric value',fontsize=9)
fig.suptitle('Continuous novelty analysis and benchmarking against a naive baseline',fontsize=13,weight='bold',y=0.98)
plt.savefig(O+'figure2_continuous_and_baseline.png',dpi=150,bbox_inches='tight')
print('saved')
