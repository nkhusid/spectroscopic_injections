This repo operates on top of the `ringdown` framework to infer remnant BH and QNM properties from ringdown signals, as well as to post-process analysis results and generate data products and figures.

## `scripts/`

To preserve the structure of this repo locally, modify the paths in `directories.py` to your own local destinations. Then use the following procedure to contribute:

1. Use `DS_inject.py` to generate damped sinusoid injections (comprised of a custom number of sinusoids, with all properties based on GW250114 and the option to vary mode amplitude ratios). Use `NR_inject.py` to generate NR injections using LVCNR formate `.h5` files (example in `data/sxs_sims/fetch_sims.ipynb`) for various remnant masses and source inclinations). The injected waveforms are saved as `.hdf5` files that populate `data/`. Several analyses are configured by default, with the option to indicate additional mode combinations to include in QNM templates. The corresponding configuration files are stored in `configs/`, with filename suffixes distinguishing between injections by indicating freely tuned knobs like post-peak SNR, mode amplitude ratio, remnant mass, inclination, etc. separated by `'_'`.

2. Upon generating the desired injections and config files, run the analyses by submitting jobs on the `rusty` cluster via `run.py`.

3. Once the sampling has completed, use `make_plots.py` (which uses the utility functions in `plotting_utils.py`) to generate summary figures comparing runs across mode combinations for a given injection. The outputs are joint remnant mass and spin posteriors over time, amplitude decay plots, LOO model comparisons across fit start time, and _joint posteriors on Kerr spectrum deviation parameters for beyond-Kerr runs (WIP)_.

## `data/`

Contains the injection waveform projected into each interferometer (`H1`, `L1`) as `.hdf5` time series, as well as the LVCNR files for the NR injections. Also gets populated with cached LOO comparison dataframes the first time `make_plots.py` is run for a given set of injection results (these are not tracked in the repo, but become available locally).

## `configs/`

Contains the configuration files for all runs.

## `notebooks/`

Contains notebooks to visualize the various DS and NR injections in the detector frame, as well as any other case-by-case investigations of injection results.
