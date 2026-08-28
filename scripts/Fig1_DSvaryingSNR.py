# import make_plots
from make_plots import *

fig, ax = plt.subplots(3, 1, figsize=(8, 6))

### top panel: DS Lorentzian plotted over PSD (Hanford)
ifo = 'H1'
psd = pd.read_hdf(str(dirs.datdir / f'bilby-NRSur7dq4_high_f_cal_{ifo}_psd_patched4kHz_1e-40.hdf5'), key=ifo)
snrs = [20, 40, 80, 160, 1000]

colls = {snr: rd.ResultCollection.from_netcdf(str(dirs.resdir / 'DS' / '220' / f'220+221_DS_GW250114Kerr_fmin10Hz_ppSNR{snr}_dfpre-0.5_dtaupre0' / 'engine' / '*' / 'result.nc')) for snr in snrs}
for coll in colls.values():
    coll.reindex_by_t0(reference_mass=coll[0].config['model']['m_min'] * 2, reference_time=t0, decimals=0)
dfs = {snr: coll.get_parameter_dataframe(ndraw=500, prng=13, progress=True) for snr, coll in colls.items()}

ax[0].loglog(psd, color='k')
dt = 1 / 4096
for snr, coll in colls.items():
    h = pd.read_hdf(coll[0].config['data']['path'][ifo], key=ifo)
    h_freq = np.fft.rfft(h.values) * dt
    fs = np.fft.rfftfreq(len(h.index.values), d=dt)
    l, = ax[0].loglog(fs, 4*fs*np.abs(h_freq)**2, lw=4)
ax[0].set_xlabel("Frequency (Hz)", fontsize=18)
ax[0].set_ylabel("PSD", fontsize=18)
# ax[0].legend()

### bottom panel: measured 221 amplitudes 
combo = '220+221'
modes = combo.split('+')
trefs_list = [-9, -3, -3, -3]
trefs = dict(zip(snrs, trefs_list))
a_dfs = {snr: {mode: get_projection(dfs[snr], mode, tref) for mode in combo.split('+')} for snr, tref in trefs.items()} 

clevs = [0.95, 0.68]

legend_handles = []
for mark, (snr, df) in enumerate(dfs.items()):
    c = f'C{mark}'

    for i, mode in enumerate(combo.split('+')):
        # if i < len(modes_true):
        shift = np.linspace(-0.15, 0.15, len(dfs))[mark]
        plot_scan(df, modes[i], clevs, c, ax[i+1], marker=filled[1], shift=shift, a_scale=1e-21)
        plot_projection(a_dfs[snr][modes[i]], [min(clevs)], ax[i+1], color=c, a_scale=1e-21)

        ax[i+1].axvline(trefs[snr]+shift, ls=':', lw=1, color=c)
        ax[i+1].set_ylabel(f'$A_{{{modes[i]}}}$ [$10^{{{int(np.log10(1e-21))}}}$]', fontsize=18)
        ax[i+1].set_xlim(-12.5, 12.5)
        ax[i+1].set_xticks(np.arange(-12, 15, 3))
        ax[i+1].tick_params(axis='both', labelsize=16, direction='in')

    legend_handles.append(Line2D([0, 0], [0, 1], color=c, marker=filled[1], 
                                            linestyle='-', linewidth=2, label=snr, 
                                            markerfacecolor=c, markeredgecolor=c))

ax[1].set_xticklabels([])
ax[2].set_xlabel('$t_> = t - t_{\\mathrm{peak}}$ [$t_{M_f}$]', fontsize=18)
ax[1].set_ylim(0, 40)
ax[2].set_ylim(0, 60)
leg = ax[2].legend(handles=legend_handles, loc=(0.5, 0.42), frameon=False, ncol=2, fontsize=13, title='post-peak SNR')

plt.savefig(str(dirs.figdir / 'paper' / 'Fig1_DSvaryingSNR.pdf'), bbox_inches='tight')