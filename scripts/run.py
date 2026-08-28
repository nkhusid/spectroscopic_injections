from pathlib import Path
import subprocess
import argparse 
import directories as dirs
import glob

def submit_runs(relpath):
    
    configdir = dirs.condir / relpath
    outputdir = dirs.resdir / relpath
    if not outputdir.exists():
        outputdir.mkdir(parents=True)

    files = glob.glob(str(configdir / '*.ini'))

    for file in files:
        outpath = outputdir / file.split(str(configdir / ''))[-1].split('.ini')[0][1:]
        print(str(outpath))
        if not outpath.exists():
            subprocess.run(['ringdown_pipe', file, '-o', str(outpath), '--time', '23:59:00', '--submit'])


def main():

    parser = argparse.ArgumentParser(description='Run all available models for a given injection.')

    parser.add_argument('--path', required=True, help='Relative path to the directory containing the config files for the injection, e.g. "DS/220".')

    parser.add_argument('--no-linesub', action='store_false', dest='linesub', default=True, help='Use injections that were not pre-processed with line subtraction before conditioning.')

    args = parser.parse_args()
    path = Path(args.path)
    if args.linesub and 'DS' not in str(path):
        path = path / 'linesub'

    submit_runs(relpath=path)

    ##### TGR runs #####

    submit_runs(relpath=path / 'tgr')

    ##### DS runs #####

    submit_runs(relpath=path / 'ds')

if __name__ == "__main__":
    main()

# module load modules/2.0-20220630 gcc/11.2.0 python/3.10 disBatch