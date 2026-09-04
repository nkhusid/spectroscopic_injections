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
import DS_inject as inj_ds

t0 = inj_ds.t0

class remnant_ds:
    m = inj_ds.m
    chi = inj_ds.chi
    # TM = rd.qnms.T_MSUN * m

##### remnant final mass and spin #####
def plot_mfcf_man(df, mf_true, cf_true, ax, legend=True, pe_df=None, **ckws):
    ''' 
    Plotting the remnant final mass and spin posteriors for a scan of analysis start times with respect to the peak. Takes a posterior dataframe containing samples (chain and draw dimensions collapsed to 1D) with columns 'm', 'chi', and 'run'. The 'run' column serves as an analysis start time label.

    df: pandas.DataFrame
        dataframe with posterior samples
    mf_true: float
        Injected final mass
    cf_true: float
        Injected final spin
    ax: matplotlib.Axes
        axes object for subplotting
    legend: bool (default is True)
        whether or not to include legend on subplot
    ckws:
        additional kwargs to pass to seaborn.kdeplot(), including 'palette'
    '''

    runs = df['run'].unique()
    palette = sns.color_palette(ckws.pop('palette', 'coolwarm'), n_colors=len(runs))
    
    for i, run in enumerate(runs):
        lw = 2 if run != 0 else 5
        sns.kdeplot(df[df['run']==run], x='m', y='chi', color=palette[i], 
                    levels=[1-0.864], common_norm=False, zorder=-1*i, alpha=0.85, linewidths=[lw], legend=False, ax=ax, **ckws)
        if pe_df is not None:
            rng = np.random.default_rng(seed=13)
            idx = rng.integers(0, len(df), size=1000)
            sns.kdeplot(pe_df.iloc[idx], x='final_mass', y='final_spin', color='k', 
                            levels=[1-0.864, 1], common_norm=False, zorder=-100, alpha=0.2, fill=True, legend=False, ax=ax, **ckws)

    ax.axvline(mf_true, ls='--', c='k', alpha=0.5, zorder=-10)
    ax.axhline(cf_true, ls='--', c='k', alpha=0.5, zorder=-10)
    
    # add custom legend
    # TM = rd.qnms.T_MSUN * mf_true
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

##### f-gamma timescans #####
def plot_fg_man(df: pd.DataFrame | rd.ResultCollection, modes, fs, gs, ax, legend=True, **ckws):

    if isinstance(df, pd.DataFrame):
        runs = df['run'].unique()
    else:
        coll = df
        runs = list(coll.idx.keys())
    palette_names = ckws.pop('palette', ['Blues', 'Greens', 'Oranges', 'Purples', 'Reds'])
    palettes = [sns.color_palette(palette, n_colors=len(runs)+1) for palette in palette_names]

    for mode in fs.keys():
        ax.scatter(x=fs[mode], y=gs[mode], color='k', marker='*', s=300, lw=0, zorder=10)
        ax.text(s=f'{mode[0]}{mode[1]}{mode[2]}', x=fs[mode], y=gs[mode]-(0.075*(max(gs.values()) - min(gs.values()))), ha='center', va='top')

    for i, run in enumerate(runs[::-1]):
        for m, mode in enumerate(modes):
            lw = 2
            if run == 0:
                lw = 4
            if isinstance(df, pd.DataFrame):
                sns.kdeplot(df[df['run'] == run], x=f'f_{mode}', y=f'g_{mode}', 
                            color=palettes[m][i], alpha=(i+1)/(len(runs)),
                            linewidths=lw, levels=[0.1], ax=ax)
            else:
                sns.kdeplot(x=coll.idx[run].posterior.f[:,:,mode].values.flatten()[::4],
                            y=coll.idx[run].posterior.g[:,:,mode].values.flatten()[::4], 
                            color=palettes[m][i], alpha=(i+1)/(len(runs)),
                            linewidths=lw, levels=[0.1], ax=ax)

    ax.set_xlabel('$f$ [Hz]')
    ax.set_ylabel('$\gamma$ [Hz]')

    frange = max(fs.values()) - min(fs.values())
    grange = max(gs.values()) - min(gs.values())
    ax.set_xlim(min(fs.values())-frange*0.25, max(fs.values())+frange*0.25)
    ax.set_ylim(0, max(gs.values())+grange*0.25)

##### mode amplitude evolution #####
def get_projection(df, mode, t0ref):
    t = np.arange(-15, 16, 3)

    TM = remnant_ds.m * rd.qnms.T_MSUN
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
            kws['lw'] = 2
        else:
            kws['capsize'] = None
            kws['lw'] = 5

        _, caps, bars = ax.errorbar(m.index+shift, m/a_scale, ((m - lohi.iloc[:,0])/a_scale, (lohi.iloc[:,1] - m)/a_scale), marker=" ", markersize=1, linestyle="", c=color, alpha=0.5*alpha_scale, **kws)
        for bar in bars:
            bar.set_capstyle('round')
        if marker is not None:
            ax.plot(m.index +shift, m/a_scale, marker=marker,  markersize=7, color=color, alpha=0.95*alpha_scale, linestyle="")
