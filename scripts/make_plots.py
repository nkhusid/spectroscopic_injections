from plotting_utils import *
import directories as dirs

import argparse

import ringdown as rd

def main():

    parser = argparse.ArgumentParser(description='Make summary plots for spectroscopic analysis of injections.')

    parser.add_argument('--path', required=True, help='Relative path to the directory containing the results for the injection, e.g. "DS/220".')

    # parser.add_argument('--ringup', required=False, default='equal', help='Morphology for the ringup of the ')

    args = parser.parse_args()

    resdir = dirs.resdir / args.path
    subdirs = [x.name for x in resdir.iterdir() if x.is_dir()]

    savedir = dirs.figdir / args.path
    if not savedir.exists():
        savedir.mkdir(parents=True)

    # Group by full suffix (everything after the 'fmin*Hz')
    groups = {}
    for name in subdirs:
        suffix = f'{name.split("Hz", 1)[1]}' if "_" in name else ''
        if suffix not in groups:
            groups[suffix] = []
        groups[suffix].append(name)
    # print(groups)

    for group, grouped_subdirs in groups.items():

        figpath = savedir / f'mchi_summary{group}.pdf'
        if not figpath.exists() and 'tgr' not in grouped_subdirs:

            ### Load all results for a given injection
            colls = {}
            dfs = {}
            # print(sorted(grouped_subdirs))
            for subdir in sorted(grouped_subdirs):
                combo = subdir.split('_')[0]
                coll = rd.ResultCollection.from_netcdf(str(resdir / subdir / 'engine' / '*' / 'result.nc'))
                colls[combo] = coll
                df = coll.get_parameter_dataframe(ndraw=500, prng=13)
                df['run'] = round((df['run'] - t0) / TM)
                # print(df['run'].unique())
                dfs[combo] = df

            fig, ax = plt.subplots(1, len(dfs), figsize=(11*1.5, 5), sharex=True, sharey=True)

            for i, (combo, df) in enumerate(dfs.items()):
                legend = True
                if i != 0:
                    legend = False
                plot_mfcf_man(df, m, chi, ax[i], legend=legend, **dict(palette='Oranges_r'))
                ax[i].set_title(f'QNM model: {combo}')

            fig.suptitle(f'DS injection: Kerr {args.path.split("/")[-1]}')

            plt.savefig(str(figpath), bbox_inches='tight')


if __name__ == "__main__":
    main()