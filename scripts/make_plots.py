from plotting_utils import *
import directories as dirs

from matplotlib.lines import Line2D
filled = list(Line2D.filled_markers)

import argparse
import copy

import ringdown as rd

def main():

    parser = argparse.ArgumentParser(description='Make summary plots for spectroscopic analysis of injections.')

    parser.add_argument('--path', required=True, help='Relative path to the directory containing the results for the injection, e.g. "DS/220".')

    parser.add_argument('--suffix', required=False, default=False, nargs='+', help='Specify injections for which to generate plots.')

    parser.add_argument('--no-mchi', action='store_false', dest='mchi', default=True, help='Generate mchi plot. Defaults to True.')

    parser.add_argument('--no-amps', action='store_false', dest='amps', default=True, help='Generate amplitude decay plot. Defaults to True.')

    parser.add_argument('--no-comp', action='store_false', dest='comp', default=True, help='Generate LOO model comparison plot. Defaults to True.')

    parser.add_argument('--no-cache', action='store_false', dest='cache', default=True, help='Do not load cached .hdf5 to generate LOO plot. Defaults to True.')

    # parser.add_argument('--ringup', required=False, default='equal', help='Morphology for the ringup of the ')

    args = parser.parse_args()
    combo_true = args.path.split('/')[-1]
    modes_true = combo_true.split('+')

    resdir = dirs.resdir / args.path
    subdirs = [x.name for x in resdir.iterdir() if x.is_dir()]

    savedir = dirs.figdir / args.path
    if not savedir.exists():
        savedir.mkdir(parents=True)

    cachedir = dirs.datdir / args.path / 'comp'
    if not cachedir.exists():
        cachedir.mkdir(parents=True)

    # Group by full suffix (everything after the 'fmin*Hz')
    groups = {}
    if not args.suffix:
        for name in subdirs:
            suffix = f'{name.split("Hz", 1)[1]}' if "_" in name else ''
            if suffix not in groups:
                groups[suffix] = []
            groups[suffix].append(name)
    else:
        for name in subdirs:
            suffix = f'{name.split("Hz", 1)[1]}' if "_" in name else ''
            if suffix[1:] in args.suffix:
                if suffix not in groups:
                    groups[suffix] = []
                groups[suffix].append(name)
    # print(groups)

    for group, grouped_subdirs in groups.items():
        print(group)

        mchi_figpath = savedir / f'mchi_summary{group}.pdf'
        amps_figpath = savedir / f'amps_summary{group}.pdf'
        comp_figpath = savedir / f'comp_summary{group}.pdf'

        cachepath = cachedir / f'loo_comp_df{group}.hdf5'

        ### FOR NON-TGR RUNS ###
        if 'tgr' not in grouped_subdirs:

            ### Load all results for a given injection ###
            colls = {}
            dfs = {}
            # print(sorted(grouped_subdirs))
            for subdir in sorted(grouped_subdirs):
                combo = subdir.split('_')[0]
                try:
                    coll = rd.ResultCollection.from_netcdf(str(resdir / subdir / 'engine' / '*' / 'result.nc'))
                    # coll.reindex_by_t0(reference_mass=m, reference_time=t0, decimals=1)
                    colls[combo] = coll
                    df = coll.get_parameter_dataframe(ndraw=500, prng=13)
                    df['run'] = round((df['run'] - t0) / TM)
                    # print(df['run'].unique())
                    dfs[combo] = df
                except (OSError, ValueError) as e:
                    ### results not yet available or being actively written to .nc file
                    print(e)

            ### MCHI SUMMARY PLOT ###
            if args.mchi:
                print('Making mchi plots...')
                fig, ax = plt.subplots(1, len(dfs), figsize=(11*1.5, 5), sharex=True, sharey=True)
                if len(dfs) == 1:
                    ax = [ax]

                for i, (combo, df) in enumerate(dfs.items()):
                    legend = True
                    if i != 0:
                        legend = False
                    plot_mfcf_man(df, m, chi, ax[i], legend=legend, **dict(palette='Oranges_r'))
                    ax[i].set_title(f'QNM model: {combo}')

                fig.suptitle(f'DS injection: Kerr {combo_true}')

                plt.savefig(str(mchi_figpath), bbox_inches='tight')

            ### AMPS SUMMARY PLOT ###
            # if args.amps:
            #     print('Making amplitude plots...')
            #     tref = 0
            #     a_true_df = {mode: get_projection(dfs[combo_true], mode, tref) for mode in modes_true}

            #     fig, ax = plt.subplots(len(modes_true), 1, figsize=(8, 4), layout='constrained', sharex=True)
            #     a_scale = 1e-21
            #     clevs = [0.9, 0.5]

            #     if type(ax) is not np.ndarray:
            #         ax = [ax]

            #     ax[len(modes_true)-1].set_xlabel('$t_> = t - t_{\\mathrm{peak}}$ [$t_{M_f}$]', fontsize=18)
            #     ax[len(modes_true)-1].set_xticks(np.arange(-12, 15, 3))

            #     c = sns.color_palette('Oranges', n_colors=1)[0]

            #     for i, mode in enumerate(modes_true):
            #         plot_projection(a_true_df[mode], clevs, ax[i])

            #     legend_handles = []
            #     for mark, (combo, df) in enumerate(dfs.items()):

            #         for i, mode in enumerate(combo.split('+')):
            #             if i < len(modes_true):
            #                 plot_scan(df, mode, clevs, c, ax[i], marker=filled[mark+1], shift=np.linspace(-0.15, 0.15, len(dfs))[mark])
            #                 ax[i].set_ylabel(f'$A_{{{i}}}$ [$10^{{{int(np.log10(a_scale))}}}$]', fontsize=18)
            #                 ax[i].set_xlim(-12.5, 12.5)
            #                 ax[i].set_ylim(0, 10)
            #                 ax[i].tick_params(axis='both', labelsize=16, direction='in')
                    
            #         legend_handles.append(Line2D([0, 0], [0, 1], color='k', marker=filled[mark+1], 
            #                                         linestyle='-', linewidth=2, label='Kerr '+combo, 
            #                                         markerfacecolor='k', markeredgecolor='k'))
                
            #     leg = ax[0].legend(handles=legend_handles, loc=(0.47, 0.52), frameon=False, ncol=2, fontsize=13)
                
            #     plt.savefig(str(amps_figpath), bbox_inches='tight')

            ### LOO SUMMARY PLOT ###
            if args.comp:
                print('Making LOO comparison plot...')

                if (cachepath.exists() and args.cache):

                    comp_df = pd.read_hdf(str(cachepath), key=group)
                
                else:

                    if args.cache:
                        print(f'Model comparison file {str(cachepath)} does not yet exist, computing LOOs (and saving to cache)...')

                    elif cachepath.exists():
                        print(f'Computing LOOs and overwriting exists model comparison file {str(cachepath)}...')

                    loo_colls = copy.deepcopy(colls)

                    for combo, coll in loo_colls.items():
                        coll.reindex_by_t0(reference_mass=m, reference_time=t0, decimals=1)

                    loo_dict = pd.DataFrame({combo: [r.idx[s].loo.elpd_loo for s in dfs[combo]['run'].unique()] for combo, r in loo_colls.items()}, index=np.array(dfs[list(dfs.keys())[0]]['run'].unique()))

                    model_dfs = []
                    for s in loo_dict.index:
                        model_df = az.compare({k: r.idx[s] for k, r in loo_colls.items()}, ic='loo', var_name='whitened_pointwise_loglike')
                        model_df['$t_> = t - t_{\\mathrm{peak}}$ [$t_{M_f}$]'] = s
                        model_dfs.append(model_df)

                    comp_df = pd.concat(model_dfs, ignore_index=False)
                    comp_df['model'] = comp_df.index
                    comp_df['elpd_diff_doub'] = comp_df['elpd_diff'] * 2
                    comp_df['dse_doub'] = comp_df['dse'] * 2

                    comp_df.to_hdf(str(cachepath), key=group, mode='w')

                fig, ax = plt.subplots()
                sns.scatterplot(comp_df, x='$t_> = t - t_{\\mathrm{peak}}$ [$t_{M_f}$]', y='elpd_diff_doub', hue='model', s=70, ax=ax)
                for i in range(len(comp_df)):
                    if comp_df.iloc[i]['rank'] != 0:
                        ax.errorbar(x=comp_df.iloc[i]['$t_> = t - t_{\\mathrm{peak}}$ [$t_{M_f}$]'], y=comp_df.iloc[i]['elpd_diff_doub'], yerr=comp_df.iloc[i]['dse_doub'], 
                                    lw=0, elinewidth=2, c=f'C{list(comp_df.index.unique()).index(comp_df.index[i])}', capsize=7, capthick=2, zorder=-1)
                ax.set_ylabel('$2 \\times \Delta \mathrm{ELPD} \sim |\Delta \chi^2|$')
                ax.axhline(0, c='gray', ls='--', zorder=-1)
                ax.grid(alpha=0.2)
                ax.set_yticks(np.arange(0, 10))
                ax.set_ylim(9, -1)
                plt.savefig(str(comp_figpath), bbox_inches='tight')

            ### AMPS SUMMARY PLOT ###
            if args.amps:
                print('Making amplitude plots...')

                trefs = {}
                if not args.comp:
                    try:
                        print('Loading cached loo comp .hdf5 file...')
                        comp_df = pd.read_hdf(str(cachepath), key=group)
                    except FileNotFoundError as e:
                        raise FileNotFoundError('Run with --comp True to generate LOO comparison DataFrame for determining amplitude projection time.') from e
                for combo in comp_df['model'].unique():
                    model_df = comp_df[comp_df['model'] == combo]
                    # ts_df = model_df[model_df['elpd_diff'] < 4]
                    ts = model_df['$t_> = t - t_{\\mathrm{peak}}$ [$t_{M_f}$]'].values
                    diffs = model_df['elpd_diff'].values
                    best = diffs < 4
                    idx = next(b for b in range(len(best)) if np.all(best[b:]))
                    if idx is None:
                        tref = max(ts)
                    else:
                        tref = ts[idx]
                    trefs[combo] = tref

                a_dfs = {combo: {mode: get_projection(dfs[combo], mode, tref) for mode in combo.split('+')} for combo, tref in trefs.items()}

                # a_true_df = {mode: get_projection(dfs[combo_true], mode, tref) for mode in modes_true}

                fig, ax = plt.subplots(max([len(combo.split('+')) for combo in a_dfs.keys()]), 1, figsize=(8, 4), layout='constrained', sharex=True)
                a_scale = 1e-21
                clevs = [0.9]

                if type(ax) is not np.ndarray:
                    ax = [ax]

                ax[len(ax)-1].set_xlabel('$t_> = t - t_{\\mathrm{peak}}$ [$t_{M_f}$]', fontsize=18)
                ax[len(ax)-1].set_xticks(np.arange(-12, 15, 3))

                legend_handles = []
                for mark, (combo, df) in enumerate(dfs.items()):
                    c = f'C{mark}'

                    for i, mode in enumerate(combo.split('+')):
                        # if i < len(modes_true):
                        shift = np.linspace(-0.15, 0.15, len(dfs))[mark]
                        plot_scan(df, mode, clevs, c, ax[i], marker=filled[mark+1], shift=shift)
                        plot_projection(a_dfs[combo][mode], clevs, ax[i], color=c)
                        ax[i].axvline(trefs[combo]+shift, ls=':', lw=1, color=c)

                        ax[i].set_ylabel(f'$A_{{{i}}}$ [$10^{{{int(np.log10(a_scale))}}}$]', fontsize=18)
                        ax[i].set_xlim(-12.5, 12.5)
                        ax[i].set_ylim(0, 10)
                        ax[i].tick_params(axis='both', labelsize=16, direction='in')
                    
                    legend_handles.append(Line2D([0, 0], [0, 1], color=c, marker=filled[mark+1], 
                                                    linestyle='-', linewidth=2, label='Kerr '+combo, 
                                                    markerfacecolor=c, markeredgecolor=c))
                
                leg = ax[0].legend(handles=legend_handles, loc=(0.47, 0.52), frameon=False, ncol=2, fontsize=13)
                
                plt.savefig(str(amps_figpath), bbox_inches='tight')


if __name__ == "__main__":
    main()