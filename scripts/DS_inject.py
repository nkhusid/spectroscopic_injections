import numpy as np

import ringdown as rd
import directories as dirs

import argparse

## GW150914 target

t0 = 1126259462.423
ra = 1.952318922
dec = -1.26967171703
psi = 0.824043851821

## Loading Kerr 220+221 fits to GW250114 for reference damped sinusoid parameters
coll = rd.ResultCollection.from_netcdf('/mnt/home/nkhusid/ceph/GW250114_review/stored_runs/AreaLawReview/220+221/start4M_stop6M_step0.5M_T0.6s_ds4_PSDsrate4kHz/engine/*/result.nc')

## Using the fit at 6 M, taking the posterior sample draw correponding to the median 221 amplitude
res = coll[-1]
p = res.posterior
imed = np.argmin(np.abs(p.a[:,:,1].values - np.median(p.a[:,:,1].values)))

## Obtain reference remnant final mass and spin measurements
chi = p.chi.values.flatten()[imed]
m = p.m.values.flatten()[imed]

def main():

    chi = p.chi.values.flatten()[imed]
    m = p.m.values.flatten()[imed]

    parser = argparse.ArgumentParser(description="Construct a damped sinusoid signal injection with parameters similar to GW250114, scaled to some indicated final mass.")

    parser.add_argument('--m', required=False, default=None, type=float, help='Reference final mass of the remnant in solar masses. Default is None, which uses the median overtone amplitude posterior draw from the GW250114 220+221 fit @ 6M.')

    parser.add_argument("--modes", required=True, nargs='+', help="Kerr modes to inject, specified as a list of strings of the form 'lmn', e.g. --modes 220 221")

    parser.add_argument("--df", required=False, default=0, type=float, help='Fractional frequency shift applied to damped sinusoid(s) before t0. Default is 0 (no shift).')

    parser.add_argument("--dtau", required=False, default=0, type=float, help='Fractional damping time shift applied to damped sinusoid(s) before t0. Default is 0 (no shift).')

    parser.add_argument("--snr", required=False, default=20, help='Target post-peak SNR of the injection. Default is 20.')

    parser.add_argument("--aratio", required=False, default=None, type=float, nargs='+', help='Custom amplitude ratio for additional damped sinusoid over the first (which is like the 220).')

    parser.add_argument("--start", required=False, default=-12, help='Start time of the analysis window relative to t0 in units of M_f. Default is -12.')

    parser.add_argument("--stop", required=False, default=12, help='Stop time of the analysis window relative to t0 in units of M_f. Default is -12.')

    parser.add_argument("--step", required=False, default=3, help='Increment for the analysis window time scan in units of M_f. Default is 3.')

    parser.add_argument('--add_temp', required=False, nargs='+', default=None, help='Additional mode combos to run.')

    parser.add_argument('--remove_temp', required=False, nargs='+', default=None, help='Remove mode combos to run.')

    parser.add_argument('--no-write', action='store_false', dest='write', default=True, help='Do not write data/config files.')

    args = parser.parse_args()

    if args.m is not None:
        m = args.m

    TM = rd.qnms.T_MSUN * m

    print(f'Damped sinusoid injection constructed from remnant properties: (M_f, chi_f) = ({m}, {chi})')
    
    ## Damped sinusoids to inject
    modes = args.modes
    modesstr = '+'.join(modes)

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

    ## Obtain other relevant damped sinusoid parameters
    a = np.array([p.a[:,:,i].values.flatten()[imed] for i in range(len(modes))])
    ellip = [p.ellip[:,:,i].values.flatten()[imed] for i in range(len(modes))]
    theta = [p.theta[:,:,i].values.flatten()[imed] for i in range(len(modes))]
    phi = [p.phi[:,:,i].values.flatten()[imed] for i in range(len(modes))]
    omega = [2*np.pi*rd.qnms.get_ftau(m, chi, l=int(mode[0]), m=int(mode[1]), n=int(mode[2]))[0] for mode in modes]
    gamma = [1/rd.qnms.get_ftau(m, chi, l=int(mode[0]), m=int(mode[1]), n=int(mode[2]))[1] for mode in modes]

    if args.aratio is not None:
        for i, scale in enumerate(args.aratio):
            a[i+1] = a[0] * float(scale)

    ############ temp_fit ###########

    f_lo, t_lo = rd.qnms.get_ftau(m, chi, l=2, m=0, n=0)
    f_hi, t_hi = rd.qnms.get_ftau(m, chi, l=4, m=4, n=1)

    ## constructing the damped sinusoid signals in each detector from GW250114-like parameters
    df_pre = args.df
    dtau_pre = args.dtau
    s = {ifo: rd.Ringdown.from_parameters(time=temp_fit.times[ifo], t0=temp_fit.start_times[ifo],
                                a=a, ellip=ellip, theta=theta, phi=phi,
                                omega=omega, g=gamma,
                                dtau_pre=np.array([dtau_pre]), 
                                df_pre=np.array([df_pre]),
                                two_sided=True,
                                ) for ifo in temp_fit.ifos}
    
    ## projecting the complex signals into the detectors
    s = {ifo: rd.Signal.project(s_i, ifo, t0=t0, ra=ra, dec=dec, psi=psi, delay=0) for ifo, s_i in s.items()}

    ## adding these signals manually instead of rd.Fit.inject()
    for i, s_i in s.items():
        temp_fit.add_data(s_i, ifo=i)
    
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

    ## constructing the damped sinusoid signals in each detector from GW250114-like parameters, scaled to the right target post-peak SNR
    df_pre = args.df
    dtau_pre = args.dtau
    s = {ifo: rd.Ringdown.from_parameters(time=fit.times[ifo], t0=fit.start_times[ifo],
                                a=a/snr_scale, ellip=ellip, theta=theta, phi=phi,
                                omega=omega, g=gamma,
                                dtau_pre=np.array(dtau_pre), 
                                df_pre=np.array([df_pre]),
                                two_sided=True,
                                ) for ifo in fit.ifos}
    
    ## projecting the complex signals into the detectors
    s = {ifo: rd.Signal.project(s_i, ifo, t0=t0, ra=ra, dec=dec, psi=psi, delay=0) for ifo, s_i in s.items()}

    outdir = dirs.datdir / 'injections' / 'DS' / modesstr
    if not outdir.exists():
        outdir.mkdir(parents=True)
    filename = f'DS_GW250114Kerr_fmin10Hz_ppSNR{args.snr}_dfpre{args.df}_dtaupre{args.dtau}'
    if args.aratio is not None:
        filename += f'_aratio{args.aratio if len(args.aratio) > 1 else args.aratio[0]}'
    if args.m is not None:
        filename += f'_Mf{int(args.m)}'
    for i, s_i in s.items():
        s_i.to_hdf(str(outdir / f'{filename}_{i}.hdf5'), key=i)

    if args.write:
        
        ## configuring the fit object
        fit.load_data({i: str(outdir / f'{filename}_{i}.hdf5') for i in fit.ifos})
        
        fit.condition_data(f_min=10, ds=1, trim=0)
        fit.load_acfs({'H1': str(dirs.datdir / 'bilby-NRSur7dq4_high_f_cal_H1_psd_patched4kHz_1e-40.hdf5'), 
                    'L1': str(dirs.datdir / 'bilby-NRSur7dq4_high_f_cal_L1_psd_patched4kHz_1e-40.hdf5')},
                    from_psd=True)
        
        fit.update_info('pipe', 
                        **{'seed': 13,
                            't0-ref': fit.info['target']['t0'],
                            'm-ref': m,
                            't0-start': args.start,
                            't0-stop': args.stop+args.step,
                            't0-step': args.step})

        fit.update_info('run',
                            **{'prng': 13,
                            'store_h_det': True})

        fit.update_info('model',
                            **{'a_scale_max': 1e-19,
                            'm_min': m/2,
                            'm_max': m*2,
                            'marginalized': True})
        
        if args.m is not None:
            fit.update_info('remnant_ds',
                            **{'mf_true': m, 'cf_true': chi})

        fit.info.pop('fake-data', None)
        fit.info['target'].pop('t0', None)
        
        configdir = dirs.condir / 'DS' / modesstr
        if not configdir.exists():
            configdir.mkdir()

        combos = ['220', '220+221', '220+210']
        if args.add_temp is not None:
            combos = list(set(combos+args.add_temp))
        if args.remove_temp is not None:
            combos = [x for x in combos if x not in args.remove_temp]
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
