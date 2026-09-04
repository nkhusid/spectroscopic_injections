import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from plotting_utils import * 

import pickle
import argparse
import copy
import matplotlib.gridspec as gridspec

from concurrent.futures import ProcessPoolExecutor, as_completed

def _compute_loo(args):
    combo, s, result = args
    return combo, s, result.loo

class Remnant(remnant_ds):
    def __init__(self, config_path):
        config = rd.utils.load_config(config_path)
        if 'remnant_nr' in config.keys():
            self.m = float(config['remnant_nr']['mf_true'])
            self.chi = float(config['remnant_nr']['cf_true'])
        elif 'remnant_ds' in config.keys():
            self.m = float(config['remnant_ds']['mf_true'])
            self.chi = float(config['remnant_ds']['cf_true'])
        else:
            self.m = remnant_ds.m
            self.chi = remnant_ds.chi

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, required=True, help="Path to subdir for injection type, e.g. 'DS/220' or 'SXS_BBH_1155/mtot330'")

    parser.add_argument('--model', nargs='+', required=True, type=str, help='Modes in QNM model whose results to plot, e.g. --model 220 210')

    parser.add_argument('--tref', required=False, default=0, type=int, help='Analysis time from which to project amplitude posterior, e.g. --tref -3. Default is --tref 0.')

    args = parser.parse_args()

    modes = args.model
    combo = '+'.join(modes)
    ifo = 'H1'

    if 'DS' in args.path:
        morph = 'dfpre-0.5_dtaupre0'
    else:
        morph = f'{args.path.split("/")[-1]}_inclined'
    inj_path = glob.glob(str(dirs.datdir / 'injections' / args.path / f'*ppSNR20_{morph}_{ifo}.hdf5'))
    inj = pd.read_hdf(inj_path[0], key=ifo)

    if 'DS' in args.path:
        injtype = 'DS_GW250114Kerr'
    else:
        injtype = f'{args.path.split("/")[0]}'

    # single_ds = rd.Result.from_netcdf(str(dirs.resdir / args.path / f'220_{injtype}_fmin10Hz_ppSNR20_{morph}' / 'engine' / f'{inj_ds.t0:.6f}' / 'result.nc'))

    run = dirs.resdir / args.path / f'{combo}_{injtype}_fmin10Hz_ppSNR20_{morph}'
    config_ref = run / 'config.ini'
    remnant = Remnant(str(config_ref))
    mref = remnant.m
    cref = remnant.chi

    coll = rd.ResultCollection.from_netcdf(str(run / 'engine' / '*' / 'result.nc'))
    coll.reindex_by_t0(reference_time=inj_ds.t0, reference_mass=mref, decimals=0)
    df = coll.get_parameter_dataframe(ndraw=4000, progress=True, prng=13)

    psd_path = coll[0].config['acf']['path'][ifo]
    fit = rd.Fit()
    fit.load_data({ifo: inj_path[0]})
    fit.set_target(inj_ds.t0, inj_ds.ra, inj_ds.dec, inj_ds.psi, duration=0.5)
    TM = rd.qnms.T_MSUN * mref

    cachewinjdir = dirs.datdir / 'injections' / args.path
    if not cachewinjdir.exists():
        cachewinjdir.mkdir(parents=True)
    cachewinj_path = cachewinjdir / f'winj_ppSNR20_{morph}.pkl'

    if cachewinj_path.exists():
        print(f'Loading cached whitened injection from {cachewinj_path}...')
        with open(cachewinj_path, 'rb') as f:
            winj = pickle.load(f)
    else:
        print('No cached whitened injection, whitening now...')

        fit.load_acfs(from_psd=True, path={ifo: psd_path})
        winj = fit.whiten({ifo: inj})

        with open(cachewinj_path, 'wb') as f:
            pickle.dump(winj, f)
        print(f'Saved whitened injection to {cachewinj_path}')

    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1])  

    ax_top = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[1, 2])

    plt.tight_layout()

    ### plotting injection
    times = (inj.index.values - fit.start_times[ifo]) / TM
    l, = ax_top.plot(times, winj[ifo], color='k', lw=4)
    # ax_top.plot(times[(times >= 0) & (times <= 0.5/TM)], single_ds.whitened_data[0], lw=5, color='gray', alpha=0.5)
    # ax_top.plot(times[(times >= 0) & (times <= 0.5/TM)], np.quantile(single_ds.whitened_templates[0,:,:], q=0.5, axis=-1), lw=2.5, color='red', alpha=0.5, ls='--')
    ax_top.axvline(0, color='k', ls=':', lw=1)
    # ax_top.axhline(0, color='red', ls=':', lw=1)
    ax_top.axvspan(-12, 12, color='gray', alpha=0.1, lw=0)
    ax_top.set_xlim(-48, 48)
    ax_top.set_xlabel('$t - t_{\mathrm{peak}}$ [$t_{M_{\mathrm{f}}}$]')
    ax_top.set_ylabel('Whitened H1 strain')

    ax_top.xaxis.set_label_coords(0.5, -0.05, transform=ax_top.transAxes)
    l.axes.spines['bottom'].set_position(('data', 0))
    l.axes.spines['top'].set_visible(False)
    l.axes.spines['right'].set_visible(False)

    ### plotting m-chi
    print('Plotting m-chi...')

    plot_mfcf_man(df, mref, cref, ax1, legend=True, **dict(palette='Oranges_r'))

    ### plotting amps
    print('Plotting amps...')

    modecol_map = {'221': 'blue',
                   '210': 'purple',
                   '330': 'pink',}
    acol = modecol_map[modes[-1]]

    clevs = [0.95, 0.68]
    a_df = get_projection(df, modes[-1], args.tref)

    shift = 0
    a_scale = 1e-21
    plot_scan(df, modes[-1], clevs, f'tab:{acol}', ax2, marker='o', shift=shift, a_scale=a_scale)
    plot_projection(a_df, [min(clevs)], ax2, color=f'tab:{acol}', a_scale=a_scale)
    ax2.axvline(args.tref+shift, ls=':', lw=1, color=f'tab:{acol}')

    ax2.set_xlabel('$t - t_{\\mathrm{peak}}$ [$t_{M_f}$]')
    ax2.set_xticks(np.arange(-12, 15, 3))
    ax2.set_xlim(-12.5, 6.5)

    ax2.set_ylabel(f'$A_{{{modes[-1]}}}$ [$10^{{{int(np.log10(a_scale))}}}$]')
    a_max = np.quantile(df[df['run'] == args.tref][f"a_{modes[-1]}"].values, 0.55) * 2
    ax2.set_ylim(0, np.ceil(a_max / a_scale))

    ### plotting loos
    print('Plotting LOOs...')

    # Loading results from all models
    loo_colls = {combo: coll}
    extra_combos = glob.glob(str(dirs.resdir / args.path / f'*_{injtype}_fmin10Hz_ppSNR20_{morph}'))
    for run_path in extra_combos:
        extra_combo = run_path.split('/')[-1].split('_')[0]
        if extra_combo not in loo_colls.keys():
            print(f'Adding {extra_combo} to model comparison...')
            coll = rd.ResultCollection.from_netcdf(f'{run_path}/engine/*/result.nc')
            coll.reindex_by_t0(reference_time=inj_ds.t0, reference_mass=mref, decimals=0)
            loo_colls[extra_combo] = coll

    cachedir = dirs.datdir / 'injections' / args.path / 'comp'
    if not cachedir.exists():
        cachedir.mkdir(parents=True)
    cache_path = cachedir / f'loo_comp_dict_ppSNR20_{morph}.pkl'

    if cache_path.exists():
        print(f'Loading cached LOO results from {cache_path}...')
        with open(cache_path, 'rb') as f:
            loo_results = pickle.load(f)

    else:
        print('No cached LOO results, computing now...')

        # Parallelizing
        max_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
        tasks = [(combo, s, r.idx[s]) for combo, r in loo_colls.items() for s in df['run'].unique()]
        loo_results = {}   # (key, value) = ((combo, s), ELPDData)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_compute_loo, t): t for t in tasks}
            for fut in as_completed(futures):
                cb, s, loo_res = fut.result()
                loo_results[(cb, s)] = loo_res

        with open(cache_path, 'wb') as f:
            pickle.dump(loo_results, f)
        print(f'Saved LOO results to {cache_path}')

    starts = df['run'].unique()
    loo_dict = pd.DataFrame({combo: [loo_results[(combo, s)].elpd_loo for s in starts] for combo in loo_colls},
                            index=np.array(starts))

    model_dfs = []
    for s in loo_dict.index:
        model_df = az.compare({combo: loo_results[(combo, s)] for combo in loo_colls}, ic='loo')
        model_df['$t - t_{\\mathrm{peak}}$ [$t_{M_f}$]'] = s
        model_dfs.append(model_df)

    comp_df = pd.concat(model_dfs, ignore_index=False)
    comp_df['model'] = comp_df.index

    palette = sns.color_palette('Greens_r', n_colors=len(comp_df['model'].unique())+1)
    markers = ['H', '^', 'X', '*']
    marker_map = dict(zip(comp_df['model'].unique(), markers))
    sns.scatterplot(comp_df, x='$t - t_{\\mathrm{peak}}$ [$t_{M_f}$]', y='elpd_loo', hue='model', style='model',
                    markers=marker_map, s=150, ax=ax3, palette=palette)
    for i in range(len(comp_df)):
        if comp_df.iloc[i]['rank'] != 0:
            _, caps, bars = ax3.errorbar(x=comp_df.iloc[i]['$t - t_{\\mathrm{peak}}$ [$t_{M_f}$]'], y=comp_df.iloc[i]['elpd_loo'], yerr=comp_df.iloc[i]['se'], lw=3.5, c=palette[list(comp_df.index.unique()).index(comp_df.index[i])], capsize=None, alpha=0.75, zorder=-1)
            for bar in bars:
                bar.set_capstyle('round')
    ax3.set_ylabel('LOO')
    # ax3.axhline(0, c='gray', ls='--', zorder=-1)
    # ax3.grid(alpha=0.2)
    # ax3.set_yticks(np.arange(0, 10))
    ax3.set_ylim(-40)

    if 'DS' in args.path:
        plt.savefig(str(dirs.figdir / 'paper' / f'inj{args.path.split("/")[-1]}_fit{combo}_unit.pdf'), bbox_inches='tight')
    else:
        plt.savefig(str(dirs.figdir / 'paper' / f'inj{args.path.replace("/", "_")}_fit{combo}_unit.pdf'), bbox_inches='tight')

if __name__ == "__main__":
    main()