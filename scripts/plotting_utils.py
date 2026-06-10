import numpy as np
import pandas as pd
import h5py
import glob

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import arviz as az
import seaborn as sns

import ringdown as rd

import directories as dirs
import DS_inject as inj

t0 = inj.t0
m = inj.m
chi = inj.chi
TM = rd.qnms.T_MSUN * m

##### remnant final mass and spin #####
def plot_mfcf_man(df, mf_true, cf_true, ax, legend=True, **ckws):
    # g, _ = coll.plot_mass_spin(prng=13, joint_kws=dict(levels=[0.864], palette='coolwarm', alpha=0.85, linewidths=[3, 1], **ckws))
    # coll.reindex_by_t0(reference_mass=mf_true, reference_time=t0, decimals=1)
    # df = coll.get_parameter_dataframe(reference_mass=mf_true, ndraw=500, prng=13)
    # hue_order = df['run'].unique()[::-1]
    runs = df['run'].unique()
    palette = sns.color_palette(ckws.pop('palette', 'coolwarm'), n_colors=len(runs))
    
    for i, run in enumerate(runs):
        lw = 2 if run != 0 else 5
        sns.kdeplot(df[df['run']==run], x='m', y='chi', color=palette[i], 
                    levels=[1-0.864], common_norm=False, zorder=-1*i, alpha=0.85, linewidths=[lw], legend=False, ax=ax, **ckws)

    ax.axvline(mf_true, ls='--', c='k', alpha=0.5, zorder=-10)
    ax.axhline(cf_true, ls='--', c='k', alpha=0.5, zorder=-10);
    
    # add custom legend
    TM = rd.qnms.T_MSUN * mf_true
    t0s = df['run'].unique()
    lines = [Line2D([0], [0], color=sns.color_palette(palette, n_colors=len(t0s))[i]) for i in range(len(t0s))]
    labels = [f'{t:.0f}' for t in t0s]
    if legend:
        leg1 = ax.legend(lines, labels, loc='lower right', title='$t - t_{\mathrm{peak}}$ [$t_{M_{\mathrm{f}}}$]', ncol=2, frameon=False);
        ax.add_artist(leg1)

        ls = [Line2D([0], [0], color='k', lw=w) for w in [3]]
        labs = ['$2\sigma$']
        leg2 = ax.legend(ls, labs, loc='upper left', frameon=False)

    ax.set_xlim(mf_true*0.5, mf_true*1.5)
    ax.set_ylim(0, 1)

    ax.set_xlabel('$M_f$ [$M_{\odot}$]')
    ax.set_ylabel('$\chi_f$')

##### mode amplitude evolution #####
def get_projection(df, mode, t0ref):
    t = np.arange(-15, 16, 3)

    TM = m * rd.qnms.T_MSUN
    tref = (t - t0ref) * TM

    ref_df = df[df['run'] == t0ref]
    a = ref_df[f'a_{mode}'].values[:,None] * np.exp(-ref_df[f'g_{mode}'].values[:,None] * tref[None,:])
    a_df = pd.DataFrame(a, columns=t)

    return a_df

def plot_projection(a_df, cis, ax, color='gray', a_scale=1e-21):
    t = np.array(a_df.columns.values)
    for ci in cis:
        lohi = a_df.apply(lambda x: az.hdi(x.values, ci)).T
        ax.fill_between(
            t, 
            lohi.iloc[:,0]/a_scale, lohi.iloc[:,1]/a_scale,
            alpha=0.1, 
            color=color, 
            zorder=-100,
            lw=0
        )

def plot_scan(df, mode, cis, color, ax, a_scale=1e-21, marker=None, alpha_scale=1, shift=0):
    kws = {}

    m = df.groupby('run')[f'a_{mode}'].apply(lambda x: x.median())
    for i, ci in enumerate(cis):
        lohi = df.groupby('run')[f'a_{mode}'].apply(lambda x: pd.Series(az.hdi(x.values, ci))).unstack(level=1)
        
        if i == 0:
            kws['capsize'] = 0
            kws['lw'] = 1
        else:
            kws['capsize'] = None
            kws['lw'] = 2.5

        ax.errorbar(m.index+shift, m/a_scale, ((m - lohi.iloc[:,0])/a_scale, (lohi.iloc[:,1] - m)/a_scale), marker=" ", markersize=1, linestyle="", c=color, alpha=0.5*alpha_scale, **kws )
        if marker is not None:
            ax.plot(m.index +shift, m/a_scale, marker=marker,  markersize=3.5, color=color, alpha=0.95*alpha_scale, linestyle="")
