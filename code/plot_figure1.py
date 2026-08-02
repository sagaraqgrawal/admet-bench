import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
O='/mnt/user-data/outputs/'
COL={'SwissADME':'#2E5B9E','pkCSM':'#C0392B','ADMETlab3.0':'#27AE60','ADMET-AI':'#E67E22'}
LBL={'SwissADME':'SwissADME','pkCSM':'pkCSM','ADMETlab3.0':'ADMETlab 3.0','ADMET-AI':'ADMET-AI'}

def parse_ci(s):
    if pd.isna(s): return (np.nan,np.nan)
    return tuple(float(x) for x in str(s).strip('[]()').split(','))

sol=pd.read_csv(O+'sc2_solubility_SCORES.csv'); sol=sol[sol.bin=='ALL']
cls={}
for f,n in [('herg_SCORES','hERG'),('bbb_SCORES','BBB'),('dili_SCORES','DILI')]:
    d=pd.read_csv(O+f+'.csv'); cls[n]=d[d.bin=='ALL']

fig=plt.figure(figsize=(14,4.6))
gs=fig.add_gridspec(1,3,wspace=0.28,width_ratios=[1.25,0.85,1.15])

# ---- A: classification AUROC with 95% CI ----
axA=fig.add_subplot(gs[0,0])
eps=['hERG','BBB','DILI']; tools=['ADMETlab3.0','ADMET-AI','pkCSM']
x=np.arange(len(eps)); w=0.26
for i,t in enumerate(tools):
    vals,los,his=[],[],[]
    for e in eps:
        r=cls[e][cls[e].tool==t]
        v=r['AUROC'].iloc[0] if len(r) and pd.notna(r['AUROC'].iloc[0]) else np.nan
        lo,hi=parse_ci(r['AUROC_CI'].iloc[0]) if len(r) and 'AUROC_CI' in r else (np.nan,np.nan)
        vals.append(v); los.append(v-lo if pd.notna(v) else np.nan); his.append(hi-v if pd.notna(v) else np.nan)
    xs=x+(i-1)*w
    ok=~pd.isna(vals)
    axA.bar(xs[ok],np.array(vals)[ok],w,color=COL[t],label=LBL[t],edgecolor='white',linewidth=0.6,zorder=3)
    axA.errorbar(xs[ok],np.array(vals)[ok],yerr=[np.array(los)[ok],np.array(his)[ok]],
                 fmt='none',ecolor='#333',elinewidth=1.1,capsize=3,zorder=4)
axA.axhline(0.5,ls='--',c='#999',lw=1,zorder=2)
axA.text(2.42,0.515,'chance',fontsize=7,color='#777',ha='right')
axA.set_xticks(x); axA.set_xticklabels(eps); axA.set_ylim(0,1.0)
axA.set_ylabel('AUROC (95% CI)')
axA.set_title('A. Discrimination on classification endpoints',fontsize=10.5,loc='left',weight='bold')
axA.legend(fontsize=7.5,loc='upper right',framealpha=0.95); axA.grid(axis='y',alpha=0.22,zorder=0)

# ---- B: solubility RMSE with CI ----
axB=fig.add_subplot(gs[0,1])
order=['SwissADME','ADMET-AI','ADMETlab3.0','pkCSM']
vals=[];los=[];his=[]
for t in order:
    r=sol[sol.tool==t]; v=r['RMSE'].iloc[0]; lo,hi=parse_ci(r['RMSE_CI'].iloc[0])
    vals.append(v); los.append(v-lo); his.append(hi-v)
xs=np.arange(len(order))
axB.bar(xs,vals,0.62,color=[COL[t] for t in order],edgecolor='white',zorder=3)
axB.errorbar(xs,vals,yerr=[los,his],fmt='none',ecolor='#333',elinewidth=1.1,capsize=3,zorder=4)
for i,v in enumerate(vals): axB.text(i,v+0.06,f'{v:.2f}',ha='center',fontsize=8)
axB.set_xticks(xs); axB.set_xticklabels([LBL[t].replace(' ','\n') for t in order],fontsize=7.5)
axB.set_ylabel('RMSE, log mol/L (95% CI)'); axB.set_ylim(0,2.0)
axB.set_title('B. Aqueous solubility error\n(lower is better)',fontsize=10.5,loc='left',weight='bold')
axB.grid(axis='y',alpha=0.22,zorder=0)

# ---- C: MCC — includes binary-only tools ----
axC=fig.add_subplot(gs[0,2])
tools_all=['SwissADME','pkCSM','ADMETlab3.0','ADMET-AI']
w2=0.2
for i,t in enumerate(tools_all):
    vals=[]
    for e in eps:
        r=cls[e][cls[e].tool==t]
        vals.append(r['MCC'].iloc[0] if len(r) and pd.notna(r['MCC'].iloc[0]) else np.nan)
    xs=x+(i-1.5)*w2; ok=~pd.isna(vals)
    axC.bar(xs[ok],np.array(vals)[ok],w2,color=COL[t],label=LBL[t],edgecolor='white',linewidth=0.6,zorder=3)
axC.axhline(0,c='#666',lw=0.9,zorder=2)
axC.set_xticks(x); axC.set_xticklabels(eps); axC.set_ylabel('Matthews correlation coefficient')
axC.set_ylim(0,0.62)
axC.set_title('C. Agreement with experiment (MCC)\nall four tools, incl. binary-only outputs',fontsize=10.5,loc='left',weight='bold')
axC.legend(fontsize=7.5,ncol=2,loc='upper right',framealpha=0.95); axC.grid(axis='y',alpha=0.22,zorder=0)

fig.suptitle('Overall external-validation performance of four freely available ADMET tools',
             fontsize=12.5,weight='bold',y=1.03)
plt.savefig(O+'figure1_overall_performance.png',dpi=200,bbox_inches='tight')
print('saved figure1_overall_performance.png')
