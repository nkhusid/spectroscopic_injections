import numpy as np
import h5py
import glob
import argparse

import ringdown as rd
import sxs

import directories as dirs

## GW150914 target
t0 = 1126259462.423
ra = 1.952318922
dec = -1.26967171703
psi = 0.824043851821

def main():

    parser = argparse.ArgumentParser(description="Construct an NR signal injection.")

    parser.add_argument("--sim", required=True, help='Name of sub-directory containing the LVCNR injection, e.g. "SXS_BBH_1155".')

    parser.add_argument("--total_mass", required=True, type=float, help='Total mass of the remnant BH.')

    iota = parser.add_mutually_exclusive_group(required=False)
    iota.add_argument("--edge-on", dest='iota', action='store_const', const=np.pi/2, help='Set inclination of the source to be ~pi/2 radians.')
    iota.add_argument("--face-on", dest='iota', action='store_const', const=0, help='Set inclination of the source to be ~0 radians.')
    iota.add_argument("--inclined", dest='iota', action='store_const', const=np.pi/3, help='Set inclination of the source to be ~pi/3 radians.')

    parser.add_argument("--snr", required=False, default=20, help='Target post-peak SNR of the injection. Default is 20.')

    parser.add_argument("--start", required=False, default=-12, help='Start time of the analysis window relative to t0 in units of M_f. Default is -12.')

    parser.add_argument("--stop", required=False, default=12, help='Stop time of the analysis window relative to t0 in units of M_f. Default is -12.')

    parser.add_argument("--step", required=False, default=3, help='Increment for the analysis window time scan in units of M_f. Default is 3.')

    parser.add_argument('--add_temp', required=False, default=None, help='Additional mode combos to run.')

    parser.add_argument('--no-linesub', action='store_false', dest='linesub', default=True, help='Do not pre-process injected waveforms with line subtraction before conditioning.')

    args = parser.parse_args()

    ## temporary rd.Fit object
    temp_fit = rd.Fit()

    ## creating empty data arrays with the appropriately delayed and discretized timestamps
    temp_fit.fake_data(duration=4,
                    prng=1234,
                    ifos=["H1", "L1"],
                    f_min=10,
                    f_samp=4096,
                    t0=t0)
    
    ## setting GW150914 target
    temp_fit.set_target(t0=t0, 
                        ra=ra, 
                        dec=dec, 
                        psi=psi, 
                        duration=0.5)
    
    fit = temp_fit.copy()

    ############ temp_fit ###########

    ## constructing the injection from an LVCNR .h5 file
    nr_path = glob.glob(str(dirs.simdir / args.sim / 'SXS_BBH_*.h5'))[0]
    # get true remnant values
    mtot = args.total_mass
    with h5py.File(nr_path, 'r') as f:
        Mflower = f.attrs['f_lower_at_1MSUN']
        mf_true = f['remnant-mass-vs-time']['Y'][-1]
        cfx = f['remnant-spinx-vs-time']['Y'][-1]
        cfy = f['remnant-spiny-vs-time']['Y'][-1]
        cfs = f['remnant-spinz-vs-time']['Y'][-1]
        cf_true = np.linalg.norm([cfx, cfy, cfs])
    mf_true *= mtot
    f_start = Mflower/mtot
    print(f'Starting frequency: {f_start} Hz')
    TM = mf_true*rd.qnms.T_MSUN
    print(mf_true, cf_true)

    f_lo, t_lo = rd.qnms.get_ftau(mf_true, cf_true, l=2, m=0, n=0)
    f_hi, t_hi = rd.qnms.get_ftau(mf_true, cf_true, l=4, m=4, n=1)

    # injection params
    dL = 440
    wf_kws = dict(
        model = 'NR_hdf5',
        mtot = mtot,
        q=1,
        # define extrinsic source parameters
        ra = ra,
        dec = dec,
        psi = psi,
        inclination = args.iota + 1e-16, ## avoid numerical issues from being exactly face-on
        dist = dL,
        # phi_ref = 2.41342424662,
        geocent_time = t0,
        f_low = f_start,
        f_ref = 20,
        window=True,
        nr_path = nr_path)
    
    # inject signal into rd.Fit object
    temp_fit.inject(**wf_kws, no_noise=True)
    
    temp_fit.condition_data(f_min=10, ds=1, trim=0)
    temp_fit.load_acfs({'H1': str(dirs.datdir / 'bilby-NRSur7dq4_high_f_cal_H1_psd_patched4kHz_1e-40.hdf5'), 
                        'L1': str(dirs.datdir / 'bilby-NRSur7dq4_high_f_cal_L1_psd_patched4kHz_1e-40.hdf5')},
                        from_psd=True)

    wd = temp_fit.whiten(temp_fit.analysis_data)
    cumsnr = {i: np.sqrt(np.cumsum(d*d)) for i, d in wd.items()}
    ppSNR = float(np.linalg.norm([cs.iloc[-1] for cs in cumsnr.values()]))
    snr_scale = ppSNR / float(args.snr)
    print(ppSNR)
    print(snr_scale)

    ############ fit ###########

    ## constructing the same NR injection, scaled to the right target post-peak SNR
    wf_kws = dict(
        model = 'NR_hdf5',
        mtot = mtot,
        q=1,
        # define extrinsic source parameters
        ra = ra,
        dec = dec,
        psi = psi,
        inclination = args.iota + 1e-16, # practically face-on
        dist = dL * snr_scale,
        # phi_ref = 2.41342424662,
        geocent_time = t0,
        f_low = f_start,
        f_ref = 20,
        window=True,
        nr_path = nr_path)
    
    # inject signal into rd.Fit object
    fit.inject(**wf_kws, no_noise=True)

    # save the detector-frame injections as .hdf5 files to later load from configs
    outdir = dirs.datdir / 'injections' / args.sim / f'mtot{int(args.total_mass)}'
    if args.linesub:
        outdir = outdir / 'linesub'
    if not outdir.exists():
        outdir.mkdir(parents=True)
    if args.iota == 0:
        inc = 'faceon'
    elif args.iota == np.pi/2:
        inc = 'edgeon'
    else:
        inc = 'inclined'
    filename = f'{args.sim}_fmin10Hz_ppSNR{args.snr}_mtot{int(args.total_mass)}_{inc}'
    for i, s_i in fit.data.items():

        if args.linesub:
            ### apply line subtraction to the detector-frame waveforms before they get conditioned to mitigate GW memory filtering artifacts

            # padding the waveform with last value of h from NR sim
            padding_time = fit.start_times[i] + 0.24
            ipad = np.argmin(np.abs(s_i.time.values - padding_time))
            s_i.iloc[ipad:] = s_i.iloc[ipad]

            # line subtraction
            tser = sxs.TimeSeries(s_i.values, time=s_i.time.values)
            tser_sub = tser.line_subtraction()

            # save line-subtracted detector-frame injection
            s_i_sub = rd.Data(tser_sub.ndarray, index=tser_sub.time, ifo=i)
            s_i_sub.to_hdf(str(outdir / f'{filename}_{i}.hdf5'), key=i)
        
        else:
            s_i.to_hdf(str(outdir / f'{filename}_{i}.hdf5'), key=i)
    
    ## configuring the fit object
    fit.load_data({i: str(outdir / f'{filename}_{i}.hdf5') for i in fit.ifos})
    
    fit.condition_data(f_min=10, ds=1, trim=0)
    fit.load_acfs({'H1': str(dirs.datdir / 'bilby-NRSur7dq4_high_f_cal_H1_psd_patched4kHz_1e-40.hdf5'), 
                   'L1': str(dirs.datdir / 'bilby-NRSur7dq4_high_f_cal_L1_psd_patched4kHz_1e-40.hdf5')},
                   from_psd=True)
    
    fit.update_info('pipe', 
                    **{'seed': 13,
                        't0-ref': fit.info['target']['t0'],
                        'm-ref': mf_true,
                        't0-start': args.start,
                        't0-stop': args.stop+args.step,
                        't0-step': args.step})

    fit.update_info('run',
                        **{'prng': 13,
                        'store_h_det': True})

    fit.update_info('model',
                        **{'a_scale_max': 1e-19,
                        'm_min': mf_true/2,
                        'm_max': mf_true*2,
                        'marginalized': True})
    
    fit.update_info('remnant_nr',
                        **{'mf_true': mf_true,
                        'cf_true': cf_true})

    fit.info.pop('fake-data', None)
    fit.info.pop('injection', None)
    fit.info['target'].pop('t0', None)
    fit.info['condition'].pop('preserve_acfs', None)
    
    configdir = dirs.condir / args.sim / f'mtot{int(args.total_mass)}'
    if args.linesub:
        configdir = configdir / 'linesub'
    if not configdir.exists():
        configdir.mkdir(parents=True)

    combos = ['220', '220+221', '220+210']
    nds = [1, 2]
    if args.add_temp is not None:
        # combos = list(set(combos.append(args.add_temp)))
        combos.append(args.add_temp)
        combos = list(set(combos))
    for combo in combos:
        fit.set_modes([(1, -2, int(mode[0]), int(mode[1]), int(mode[2])) for mode in combo.split('+')])
        fit.update_info('model', **{'modes': str(fit.modes)})
        fit.to_config(str(configdir / f'{combo}_{filename}.ini'))

        fitmodes = combo.split('+')
        if len(fitmodes) > 1:
            df_min, df_max, dg_min, dg_max = (np.zeros(np.array(fitmodes).shape) for _ in range(4))
            for i in range(len(fitmodes)):
                if i > 0:
                    df_min[i] = dg_min[i] = -0.8
                    df_max[i] = dg_max[i] = 0.8
            tgrfit = fit.copy()
            tgrfit.update_info('model', **{'df_min': list(df_min),
                                        'df_max': list(df_max),
                                        'dg_min': list(dg_min),
                                        'dg_max': list(dg_max)})
            tgrdir = configdir / 'tgr'
            if not tgrdir.exists():
                tgrdir.mkdir(parents=True)
            tgrcombo = combo.replace('+', '+d')
            tgrfit.to_config(str(tgrdir / f'{tgrcombo}_{filename}.ini'))
        
        dsfit = fit.copy()
        dsfit.update_info('model', **{'modes': len(fitmodes),
                                      'f_min': f_lo,
                                      'f_max': f_hi,
                                      'g_min': 1/t_lo/3,
                                      'g_max': 1/t_hi*1.5,
                                      'mode_ordering': 'f'})
        dsdir = configdir / 'ds'
        if not dsdir.exists():
            dsdir.mkdir(parents=True)
        dsfit.to_config(str(dsdir / f'{len(fitmodes)}DS_{filename}.ini'))


if __name__ == "__main__":
    main()
