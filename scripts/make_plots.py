from plotting_utils import *
import directories as dirs

from matplotlib.lines import Line2D
filled = list(Line2D.filled_markers)

import argparse

import ringdown as rd

def main():

    parser = argparse.ArgumentParser(description='Make summary plots for spectroscopic analysis of injections.')

    parser.add_argument('--path', required=True, help='Relative path to the directory containing the results for the injection, e.g. "DS/220".')

    parser.add_argument('--suffix', required=False, default=False, nargs='+', help='Specify specific injections for which to generate plots.')

    # parser.add_argument('--ringup', required=False, default='equal', help='Morphology for the ringup of the ')

    args = parser.parse_args()
    combo_true = args.path.split('/')[-1]
    modes_true = combo_true.split('+')

    resdir = dirs.resdir / args.path
    subdirs = [x.name for x in resdir.iterdir() if x.is_dir()]

    savedir = dirs.figdir / args.path
    if not savedir.exists():
        savedir.mkdir(parents=True)

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
                    colls[combo] = coll
                    df = coll.get_parameter_dataframe(ndraw=500, prng=13)
                    df['run'] = round((df['run'] - t0) / TM)
                    # print(df['run'].unique())
                    dfs[combo] = df
                except (OSError, ValueError) as e:
                    ### results not yet available or being actively written to .nc file
                    print(e)

            ### MCHI SUMMARY PLOT ###
            if (not mchi_figpath.exists()) or (args.suffix is not False):
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
            if (not amps_figpath.exists()) or (args.suffix is not False):
                print('Making amplitude plots...')
                tref = 0
                a_true_df = {mode: get_projection(dfs[combo_true], mode, tref) for mode in modes_true}

                fig, ax = plt.subplots(len(modes_true), 1, figsize=(8, 4), layout='constrained', sharex=True)
                a_scale = 1e-21
                clevs = [0.9, 0.5]

                if type(ax) is not np.ndarray:
                    ax = [ax]

                ax[len(modes_true)-1].set_xlabel('$t_> = t - t_{\\mathrm{peak}}$ [$t_{M_f}$]', fontsize=18)
                ax[len(modes_true)-1].set_xticks(np.arange(-12, 15, 3))

                c = sns.color_palette('Oranges', n_colors=1)[0]

                for i, mode in enumerate(modes_true):
                    plot_projection(a_true_df[mode], clevs, ax[i])

                legend_handles = []
                for mark, (combo, df) in enumerate(dfs.items()):

                    for i, mode in enumerate(combo.split('+')):
                        if i < len(modes_true):
                            plot_scan(df, mode, clevs, c, ax[i], marker=filled[mark+1], shift=np.linspace(-0.15, 0.15, len(dfs))[mark])
                            ax[i].set_ylabel(f'$A_{{{i}}}$ [$10^{{{int(np.log10(a_scale))}}}$]', fontsize=18)
                            ax[i].set_xlim(-12.5, 12.5)
                            ax[i].set_ylim(0, 10)
                            ax[i].tick_params(axis='both', labelsize=16, direction='in')
                    
                    legend_handles.append(Line2D([0, 0], [0, 1], color='k', marker=filled[mark+1], 
                                                 linestyle='-', linewidth=2, label='Kerr '+combo, 
                                                 markerfacecolor='k', markeredgecolor='k'))
                
                leg = ax[0].legend(handles=legend_handles, loc=(0.47, 0.52), frameon=False, ncol=2, fontsize=13)
                
                plt.savefig(str(amps_figpath), bbox_inches='tight')


if __name__ == "__main__":
    main()