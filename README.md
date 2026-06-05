This repo operates on top of the `ringdown` framework to infer remnant BH and QNM properties from ringdown signals, as well as to post-process analysis results and generate data products and figures.

## `scripts/`

To preserve the structure of this repo locally, modify the paths in `directories.py` to your own local destinations. Then use the following procedure to contribute:

1. Use `DS_inject.py` to generate damped sinusoid injections (comprised of a custom number of sinusoids, with all properties based on GW250114 and the option to vary mode amplitude ratios). The injected waveforms are saved as `.hdf5` files that populate `data/`. Several analyses or configured by default, with the option to indicate additional mode combinations to include in QNM templates, and the corresponding configuration files are stored in `configs/`.

2. Upon generating the desired injections and config files, run the analyses by triggering jobs on the `rusty` cluster via `run.py`.

3. Once the sampling has completed, use `make_plots.py` (which uses the utility functions in `plotting_utils.py`) to generate summary figures comparing runs across mode combinations for a given injection. This is **WiP**, with more summary figures to come (mode amplitude and phase evolution, frequency-damping rate joint posteriors, LOO model comparison, etc).

## `data/`

Contains the injection waveform projected into each interferometer (`H1`, `L1`) as `.hdf5` time series.

## `configs/`

Contains the configuration files for all runs.
