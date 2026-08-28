# import make_plots
from make_plots import *

parser = argparse.ArgumentParser()

parser.add_argument('--inj-modes', required=True, nargs='+', help='Modes in DS injection e.g. --inj-modes 220 221')

parser.add_argument('--fit-modes', required=False, default=None, nargs='+', help='Modes in QNM model whose results to plot. e.g. --fit-modes 220 221')

# parser.add_argument('--snrs', required=False, default=None, type=int, nargs='+', help='Post-peak SNRs of injection to plot. e.g. --snrs 20 40 80')

def parse_key_value(s):
        try:
            k, v = s.split('=', 1)
            return k, float(v)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Expected KEY=float, got: {s}")

parser.add_argument('--proj-time', required=True, nargs='+', type=parse_key_value, help='Manual reference projection time for different post-peak SNRs, passed as key-value pairs with mode combo keys and t_> values. e.g. --proj-time 20=-3 40=0 80=3')

parser.add_argument('--no-amps', required=False, action='store_false', default=True, dest='amps')

args = parser.parse_args()

trefs = dict(args.proj_time)
snrs = trefs.keys()

injmodes = args.inj_modes
injcombo = '+'.join(injmodes)

fitmodes = args.fit_modes if args.fit_modes is not None else injmodes
fitcombo = '+'.join(fitmodes)

fig, ax = plt.subplots(3, 1, figsize=(8, 6))

### top panel: DS Lorentzian plotted over PSD (Hanford)
ifo = 'H1'
psd = pd.read_hdf(str(dirs.datdir / f'bilby-NRSur7dq4_high_f_cal_{ifo}_psd_patched4kHz_1e-40.hdf5'), key=ifo)
# snrs = args.snrs

ax[0].loglog(psd, color='k')
dt = 1 / 4096
for snr in snrs:
    h = pd.read_hdf(str(dirs.datdir / 'injections' / 'DS' / injcombo / f'DS_GW250114Kerr_fmin10Hz_ppSNR{snr}_dfpre-0.5_dtaupre0_{ifo}.hdf5'), key=ifo)
    h_freq = np.fft.rfft(h.values) * dt
    fs = np.fft.rfftfreq(len(h.index.values), d=dt)
    l, = ax[0].loglog(fs, 4*fs*np.abs(h_freq)**2, lw=4)
ax[0].set_xlim(1, 2048)
ax[0].set_xlabel("Frequency (Hz)", fontsize=18)
ax[0].set_ylabel("PSD", fontsize=18)
# ax[0].legend()

### bottom panel: measured mode amplitudes 
if args.amps:
    colls = {snr: rd.ResultCollection.from_netcdf(str(dirs.resdir / 'DS' / injcombo / f'{fitcombo}_DS_GW250114Kerr_fmin10Hz_ppSNR{snr}_dfpre-0.5_dtaupre0' / 'engine' / '*' / 'result.nc')) for snr in snrs}
    for coll in colls.values():
        coll.reindex_by_t0(reference_mass=coll[0].config['model']['m_min'] * 2, reference_time=t0, decimals=0)
    dfs = {snr: coll.get_parameter_dataframe(ndraw=500, prng=13, progress=True) for snr, coll in colls.items()}

    # trefs_list = -3 * np.ones(len(snrs))
    # trefs = dict(zip(snrs, trefs_list))
    a_dfs = {snr: {mode: get_projection(dfs[snr], mode, tref) for mode in fitcombo.split('+')} for snr, tref in trefs.items()} 

    clevs = [0.95, 0.68]

    legend_handles = []
    for mark, (snr, df) in enumerate(dfs.items()):
        c = f'C{mark}'

        for i, mode in enumerate(fitmodes):
            # if i < len(modes_true):
            shift = np.linspace(-0.15, 0.15, len(dfs))[mark]
            plot_scan(df, mode, clevs, c, ax[i+1], marker=filled[1], shift=shift, a_scale=1e-21)
            plot_projection(a_dfs[snr][mode], [min(clevs)], ax[i+1], color=c, a_scale=1e-21)

            ax[i+1].axvline(trefs[snr]+shift, ls=':', lw=1, color=c)
            ax[i+1].set_ylabel(f'$A_{{{mode}}}$ [$10^{{{int(np.log10(1e-21))}}}$]', fontsize=18)
            ax[i+1].set_xlim(-12.5, 12.5)
            ax[i+1].set_xticks(np.arange(-12, 15, 3))
            ax[i+1].tick_params(axis='both', labelsize=16, direction='in')
            ax[i+1].set_yscale('log')

        legend_handles.append(Line2D([0, 0], [0, 1], color=c, marker=filled[1], 
                                                linestyle='-', linewidth=2, label=snr, 
                                                markerfacecolor=c, markeredgecolor=c))

        

    ax[1].set_xticklabels([])
    ax[2].set_xlabel('$t_> = t - t_{\\mathrm{peak}}$ [$t_{M_f}$]', fontsize=18)
    ax[1].set_ylim(0.7, 100)
    ax[2].set_ylim(0.3, 100)
    leg = ax[2].legend(handles=legend_handles, loc=(0.45, 0.57), frameon=False, ncol=3, fontsize=13, title='post-peak SNR')

plt.savefig(str(dirs.figdir / 'paper' / f'Fig1_DSvaryingSNR_inj{injcombo}_fit{fitcombo}.pdf'), bbox_inches='tight')