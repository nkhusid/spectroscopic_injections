import numpy as np
import pandas as pd
import h5py
import glob

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import arviz as az
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import ringdown as rd

import directories as dirs
import DS_inject as inj_ds

t0 = inj_ds.t0

class remnant_ds:
    m = inj_ds.m
    chi = inj_ds.chi
    # TM = rd.qnms.T_MSUN * m

##### remnant final mass and spin #####
def plot_mfcf_man(df, mf_true, cf_true, ax, legend=True, pe_df=None, palette='coolwarm', **cbar_kws):
    ''' 
    Plotting the remnant final mass and spin posteriors for a scan of analysis start times with respect to the peak. Takes a posterior dataframe containing samples (chain and draw dimensions collapsed to 1D) with columns 'm', 'chi', and 'run'. The 'run' column serves as an analysis start time label.

    df: pandas.DataFrame
        dataframe with posterior samples
    mf_true: float
        Injected final mass
    cf_true: float
        Injected final spin
    ax: matplotlib.Axes
        axes object for subplotting
    legend: bool (default is True)
        whether or not to include legend on subplot
    cbar_kws:
        additional kwargs to pass to the colorbar, including 'cbar_label' and 'cbar_bottom'
    '''

    runs = df['run'].unique()

    colors = palette
    if isinstance(colors, str):
        palette = sns.color_palette(colors, n_colors=len(runs))
    else:
        palette = colors

    for i, run in enumerate(runs):
        lw = 2 if run != 0 else 5
        sns.kdeplot(df[df['run']==run], x='m', y='chi', color=palette[i], 
                    levels=[1-0.864], common_norm=False, zorder=-1*i, alpha=0.85, linewidths=[lw], legend=False, ax=ax)
        if pe_df is not None:
            rng = np.random.default_rng(seed=13)
            idx = rng.integers(0, len(df), size=1000)
            sns.kdeplot(pe_df.iloc[idx], x='final_mass', y='final_spin', color='k', 
                            levels=[1-0.864, 1], common_norm=False, zorder=-100, alpha=0.2, fill=True, legend=False, ax=ax)

    ax.axvline(mf_true, ls='--', c='k', alpha=0.5, zorder=-10)
    ax.axhline(cf_true, ls='--', c='k', alpha=0.5, zorder=-10)
    
    # add custom legend
    # TM = rd.qnms.T_MSUN * mf_true
    t0s = df['run'].unique()
    lines = [Line2D([0], [0], color=sns.color_palette(palette, n_colors=len(t0s))[i]) for i in range(len(t0s))]
    labels = [f'{t:.0f}' for t in t0s]
    if legend:
        # leg1 = ax.legend(lines, labels, loc='lower right', title='$t - t_{\mathrm{peak}}$ [$t_{M_{\mathrm{f}}}$]', ncol=2, frameon=False);
        # ax.add_artist(leg1)
        cmap = ListedColormap(palette)
        norm = BoundaryNorm(np.arange(len(runs)+1), len(runs))
    
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
    
        cax = ax.inset_axes([0.53, cbar_kws.pop('cbar_bottom', 0.2), 0.45, 0.04])  # [left, bottom, width, height]
    
        cbar = ax.figure.colorbar(
            sm,
            cax=cax,
            ticks=(np.arange(len(runs)) + 0.5)[::2],
            fraction=0.046,
            pad=0.04,
            orientation='horizontal'
        )
    
        cbar.ax.set_xticklabels(runs.astype(int)[::2])
        if not cbar_kws.pop('label', True):
            cbar.ax.tick_params(axis="x", which="both", labelbottom=False)
        cbar.ax.minorticks_off()
        cbar.set_label(cbar_kws.pop('cbar_label', '$t - t_{\mathrm{peak}}$ [$t_{M_{\mathrm{f}}}$]'), fontsize=12, labelpad=5)
        cbar.ax.xaxis.set_label_position('top')

    ls = [Line2D([0], [0], color='k', lw=w) for w in [3]]
    labs = ['$2\sigma$']
    leg2 = ax.legend(ls, labs, loc='upper left', frameon=False)

    ax.set_xlim(mf_true*0.5, mf_true*1.5)
    ax.set_ylim(0, 1)

    ax.set_xlabel('$M_f$ [$M_{\odot}$]')
    ax.set_ylabel('$\chi_f$')

##### shared start-time colorbar #####
def _resize_fig_height(fig, delta_in, upper_axes):
    '''Change the figure height by `delta_in` inches, adding/removing the space just below
    `upper_axes`: every axes keeps its size in inches, axes in `upper_axes` keep their distance to
    the top edge, all other axes keep their distance to the bottom edge.'''
    W, H = fig.get_size_inches()
    H2 = H + delta_in
    for ax in fig.axes:
        p = ax.get_position()
        y0 = (p.y0*H + delta_in)/H2 if ax in upper_axes else p.y0*H/H2
        ax.set_position([p.x0, y0, p.width, p.height*H/H2])
    fig.set_size_inches(W, H2)


def add_shared_tbar(fig, runs, palette, ax_ref, ax_col=None, t_range=(-12, 12), dt=None, text=None, **kws):
    '''
    Add one model's start times to a single horizontal start-time bar shared by every call on `fig`.

    Each call adds a strip of colored cells (one per start time in `runs`, colored by `palette`).
    A strip goes in the first row whose occupied time range it does not overlap; otherwise a new
    row is added below. The whole bar is re-laid out after every call.

    Style (chosen automatically unless 'tbar_style' is given):
      'cropped' (>= 2 strips that all cover the identical span): the bar is cropped to that span
          and each model's `text` is written left of its row.
      'full' (otherwise): the bar spans `t_range` (widened if a strip extends beyond it) and each
          model's `text` is written above a bracket over its times; models with the identical span
          share a bracket, labels stacked in row order.
    Placement ('tbar_place'):
      'inset' (default): inside `ax_ref`, top-left corner of the bar group at 'inset_anchor'.
          No extra figure height is added (any added by an earlier 'between' layout is removed).
      'between': between `ax_ref` and the panels below it, horizontally aligned with `ax_col` if
          given (otherwise with `ax_ref`'s data x-coordinates); the figure height is adjusted so
          the gap between the rows just fits the bar.
    The tick labels read  t_> = <ticks> [t_Mf], with the prefix and unit on the tick-label line.

    fig: matplotlib.Figure
    runs: array of analysis start times plotted by this model
    palette: sequence of colors, same length/order as `runs`
    ax_ref: matplotlib.Axes of the strain panel (the bar sits inside it, or below it)
    ax_col: matplotlib.Axes whose horizontal extent the bar matches with 'between' placement (optional)
    t_range: (float, float)
        Start-time range the bar spans in 'full' style (set when the bar is created).
    dt: float
        Spacing between start times (cell width). Defaults to the smallest spacing in `runs`.
    text: str
        Model label (row label in 'cropped' style, bracket label in 'full' style).
    kws (read when the bar is created):
        'tbar_style': 'auto' (default), 'cropped', or 'full'
        'tbar_place': 'inset' (default) or 'between'
        'tick_prefix', 'tick_unit': text left/right of the tick labels
                                    (defaults '$t_> = $', '[$t_{M_{\mathrm{f}}}$]'; '' for none)
        'tick_step': spacing of tick marks (default 3)
        'tick_outline': white outline around the tick labels, prefix and unit, so they stay
                        readable over the strain curve (default True)
        'text_size': font size of the model labels (default 9)
        'text_bold': boldface model labels (default True)
        'inset_row_height': height of one row in inches, 'inset' placement (default 0.11)
        'inset_cell_width': width of one start-time cell in inches, 'inset' placement (default 0.25)
        'inset_anchor': (x, y) in `ax_ref` axes coordinates of the top-left corner of the inset
                        bar group, labels included (default (0.01, 0.97) in 'full' style,
                        (0.03, 0.97) in 'cropped' style, whose row labels sit by the y-axis)
        'row_height': height of one row in inches, 'between' placement (default 0.06)
        'pad_top', 'pad_bottom': space in inches above the bar's labels / below its tick labels,
                                 'between' placement (defaults 0.03, 0.06)
    kws (per call):
        'text_color': color of `text` (default: middle color of `palette`)
    '''
    runs = np.asarray(runs, dtype=float)
    if dt is None:
        dt = np.min(np.diff(np.sort(runs))) if len(runs) > 1 else 1.0

    state = getattr(fig, '_shared_tbar', None)
    if state is None:
        cax = fig.add_axes([0, 0, 1, 0.01])
        cax.set_yticks([])
        cax.tick_params(axis='x', which='both', direction='out', length=2.5, labelsize=10, pad=1.5)
        cax.minorticks_off()
        cax.set_facecolor('none')
        for sp in cax.spines.values():
            sp.set_linewidth(0.6)
        state = dict(ax=cax, rows=[], strips=[], label_artists=[], added_in=0.0,
                     t_range=t_range, ax_col=ax_col, dt=dt,
                     style=kws.get('tbar_style', 'auto'), place=kws.get('tbar_place', 'inset'),
                     tick_prefix=kws.get('tick_prefix', '$t_> = $'),
                     tick_unit=kws.get('tick_unit', '[$t_{M_{\\mathrm{f}}}$]'),
                     tick_step=kws.get('tick_step', 3), text_size=kws.get('text_size', 9),
                     tick_outline=kws.get('tick_outline', True),
                     text_bold=kws.get('text_bold', True),
                     row_height=kws.get('row_height', 0.06),
                     pad_top=kws.get('pad_top', 0.03), pad_bottom=kws.get('pad_bottom', 0.06),
                     inset_row_height=kws.get('inset_row_height', 0.11),
                     inset_cell_width=kws.get('inset_cell_width', 0.25),
                     inset_anchor=kws.get('inset_anchor', None))
        fig._shared_tbar = state

    cax = state['ax']
    rows = state['rows']
    span = (runs.min() - dt/2, runs.max() + dt/2)

    # first row where this strip does not overlap an existing one
    for r, occupied in enumerate(rows):
        if all(span[1] <= a + 1e-9 or span[0] >= b - 1e-9 for a, b in occupied):
            row = r
            occupied.append(span)
            break
    else:
        row = len(rows)
        rows.append([span])

    # edges in each cell's own color: no seams between adjacent start times
    cax.bar(runs, 1, width=dt, bottom=row, color=palette, edgecolor=palette, linewidth=0.3, align='center')
    cax.set_ylim(len(rows), 0)
    state['strips'].append(dict(span=span, row=row, text=text,
                                color=kws.get('text_color', palette[len(palette)//2])))

    _layout_shared_tbar(fig, state, ax_ref)
    return cax


def _layout_shared_tbar(fig, state, ax_ref):
    '''Choose the style, set limits/ticks, draw the labels, and position the bar.'''
    cax, dt = state['ax'], state['dt']
    for a in state['label_artists']:
        a.remove()
    state['label_artists'] = []

    spans = [s['span'] for s in state['strips']]
    same_span = len(spans) > 1 and all(np.allclose(s, spans[0]) for s in spans)
    style = state['style'] if state['style'] != 'auto' else ('cropped' if same_span else 'full')
    state['style_used'] = style

    if style == 'cropped':   # crop to the plotted times
        lo, hi = min(a for a, _ in spans), max(b for _, b in spans)
    else:                 # full t_range, widened if a strip extends beyond it
        lo = min([state['t_range'][0] - dt/2] + [a for a, _ in spans])
        hi = max([state['t_range'][1] + dt/2] + [b for _, b in spans])
    cax.set_xlim(lo, hi)
    # thin white lines separating the rows (models)
    for r in range(1, len(state['rows'])):
        state['label_artists'].append(cax.axhline(r, color='white', lw=0.6))
    step = state['tick_step']
    ticks = np.arange(np.ceil((lo + dt/2)/step)*step, hi - dt/2 + 1e-9, step)
    cax.set_xticks(ticks)
    cax.set_xticklabels([f'${t:.0f}$' for t in ticks])

    # provisional size (tick-label positions depend on the bar's width)
    W, H = fig.get_size_inches()
    p = ax_ref.get_position()
    if state['place'] == 'inset':
        w = (hi - lo)/dt*state['inset_cell_width']/W
        h = state['inset_row_height']*len(state['rows'])/H
    else:
        if state['ax_col'] is not None:
            w = state['ax_col'].get_position().width
        else:
            to_fig = ax_ref.transData + fig.transFigure.inverted()
            w = to_fig.transform((hi, 0))[0] - to_fig.transform((lo, 0))[0]
        h = state['row_height']*len(state['rows'])/H
    cax.set_position([p.x0, p.y0, w, h])

    if style == 'cropped':
        _draw_tbar_row_labels(fig, state)
    else:
        _draw_tbar_brackets(fig, state)
    _draw_tbar_tick_text(fig, state)

    if state['place'] == 'inset':
        _place_shared_tbar_inset(fig, state, ax_ref)
    else:
        _place_shared_tbar_between(fig, state, ax_ref)


def _draw_tbar_tick_text(fig, state):
    '''Write state['tick_prefix'] left of the first tick label and state['tick_unit'] right of the
    last one, on the same baseline as the tick labels.

    Placement uses the tick labels' own transform (tick x in data coords, fixed offset below the
    bar) plus font metrics of the label strings, not measured window extents (which can be stale
    before a draw), so the text stays aligned wherever the bar is moved afterwards.'''
    import matplotlib.transforms as mtransforms
    cax = state['ax']
    ticks = cax.xaxis.get_major_ticks()
    locs = cax.get_xticks()
    pairs = [(tk, x) for tk, x in zip(ticks, locs) if tk.label1.get_text()]
    if not pairs:
        return
    # white outline behind each tick label (a white copy drawn underneath)
    for tk, x in pairs:
        _tbar_halo(cax, state, x, 0, tk.label1.get_text(), tk.label1.get_transform(),
                   tk.label1.get_ha(), tk.label1.get_va(), tk.label1.get_fontsize())
    if not (state['tick_prefix'] or state['tick_unit']):
        return
    renderer = fig.canvas.get_renderer()
    usetex = plt.rcParams['text.usetex']

    def metrics(text_artist):
        '''(half width, ascent above baseline) of a tick label, in inches.'''
        s = text_artist.get_text()
        ismath = 'TeX' if usetex else ('$' in s)
        w, h, d = renderer.get_text_width_height_descent(s, text_artist.get_fontproperties(), ismath=ismath)
        return 0.5*w/fig.dpi, (h - d)/fig.dpi

    gap = 0.05
    (tk0, x0), (tk1, x1) = pairs[0], pairs[-1]
    fs = tk0.label1.get_fontsize()
    for tk, x, text, sign, ha in [(tk0, x0, state['tick_prefix'], -1, 'right'),
                                  (tk1, x1, state['tick_unit'], 1, 'left')]:
        if not text:
            continue
        half_w, ascent = metrics(tk.label1)
        # tick labels are top-aligned; their baseline is `ascent` below that line
        dy = -ascent if tk.label1.get_va() == 'top' else 0.0
        trans = tk.label1.get_transform() + \
            mtransforms.ScaledTranslation(sign*(half_w + gap), dy, fig.dpi_scale_trans)
        _tbar_halo(cax, state, x, 0, text, trans, ha, 'baseline', fs)
        state['label_artists'].append(cax.text(x, 0, text, transform=trans, ha=ha, va='baseline',
                                               fontsize=fs, clip_on=False))


def _tbar_halo(cax, state, x, y, text, transform, ha, va, fontsize):
    '''White outline for a piece of text on the tick-label line, drawn as a separate white copy
    underneath it. (Putting path effects on the text itself makes usetex text render slightly
    smaller than its measured size, which shrinks the labels and breaks the prefix/unit spacing;
    here only the invisible-ish white copy goes through that path.)'''
    if not state['tick_outline']:
        return
    import matplotlib.patheffects as pe
    t = cax.text(x, y, text, transform=transform, ha=ha, va=va, fontsize=fontsize, color='white',
                 clip_on=False, zorder=1.4,   # below the axis (tick labels) and other text
                 path_effects=[pe.withStroke(linewidth=4, foreground='white')])
    state['label_artists'].append(t)


def _tbar_label(text, state):
    '''Model label text; with usetex, boldface needs \\textbf (fontweight is ignored by LaTeX).'''
    if state['text_bold'] and plt.rcParams['text.usetex']:
        return r'\textbf{%s}' % text
    return text


def _draw_tbar_row_labels(fig, state):
    '''Model labels to the left of their rows ('cropped' style).'''
    import matplotlib.transforms as mtransforms
    cax = state['ax']
    trans = mtransforms.blended_transform_factory(cax.transAxes, cax.transData) + \
        mtransforms.ScaledTranslation(-0.04, 0, fig.dpi_scale_trans)
    for s in state['strips']:
        if s['text']:
            t = cax.text(0, s['row'] + 0.5, _tbar_label(s['text'], state), transform=trans, ha='right',
                         va='center', fontsize=state['text_size'], color=s['color'], clip_on=False,
                         fontweight='bold' if state['text_bold'] else 'normal')
            state['label_artists'].append(t)


def _draw_tbar_brackets(fig, state):
    '''Model brackets and labels above the bar ('full' style). Vertical positions are in inches
    above the bar's top edge, so they stay fixed as rows are added or the figure is resized.'''
    import matplotlib.transforms as mtransforms
    cax = state['ax']

    # models covering the identical span share a bracket
    groups = []
    for s in state['strips']:
        if not s['text']:
            continue
        for g in groups:
            if np.allclose(g['span'], s['span']):
                g['labels'].append((s['row'], s['text'], s['color']))
                break
        else:
            groups.append(dict(span=s['span'], labels=[(s['row'], s['text'], s['color'])]))
    if not groups:
        return

    gap, bh = 0.03, 0.035            # inches: bar top -> bracket feet, bracket height
    line_h = state['text_size']*1.25/72
    inset = 0.1*state['dt']          # keeps adjacent brackets visibly separate

    # brackets over disjoint spans share a level; overlapping ones go on higher levels
    levels = []
    for g in sorted(groups, key=lambda g: g['span'][0]):
        for L, occ in enumerate(levels):
            if all(g['span'][1] <= a + 1e-9 or g['span'][0] >= b - 1e-9 for a, b in occ):
                occ.append(g['span'])
                g['level'] = L
                break
        else:
            g['level'] = len(levels)
            levels.append([g['span']])
    heights = [gap + bh + 0.01 + max(len(g['labels']) for g in groups if g['level'] == L)*line_h
               for L in range(len(levels))]
    base = np.concatenate([[0], np.cumsum(heights)])

    # y in inches above the top of the bar, x in data (start-time) coordinates
    ytrans = fig.dpi_scale_trans + mtransforms.ScaledTranslation(0, 1, cax.transAxes)
    trans = mtransforms.blended_transform_factory(cax.transData, ytrans)
    for g in groups:
        x0, x1 = g['span'][0] + inset, g['span'][1] - inset
        y0 = base[g['level']] + gap
        y1 = y0 + bh
        line = Line2D([x0, x0, x1, x1], [y0, y1, y1, y0], transform=trans, color='k', lw=0.8,
                      clip_on=False, solid_joinstyle='miter')
        cax.add_artist(line)
        state['label_artists'].append(line)
        # labels stacked so their top-to-bottom order matches the rows' order
        for k, (row, txt, col) in enumerate(sorted(g['labels'], key=lambda l: -l[0])):
            t = cax.text(0.5*(x0 + x1), y1 + 0.01 + k*line_h, _tbar_label(txt, state), transform=trans,
                         ha='center', va='bottom', fontsize=state['text_size'], color=col, clip_on=False,
                         fontweight='bold' if state['text_bold'] else 'normal')
            state['label_artists'].append(t)


def _place_shared_tbar_inset(fig, state, ax_ref):
    '''Inset the bar group inside ax_ref, top-left corner of the group (labels included) at
    state['inset_anchor'] in ax_ref axes coordinates; remove any gap added by 'between' placement.

    Text extents are measured before any resize (they are rigid in inches); re-measuring text
    right after a resize can return stale extents.'''
    cax = state['ax']
    renderer = fig.canvas.get_renderer()
    to_fig = fig.transFigure.inverted()
    W, H = fig.get_size_inches()

    lo, hi = cax.get_xlim()
    w_in = (hi - lo)/state['dt']*state['inset_cell_width']
    h_in = state['inset_row_height']*len(state['rows'])
    p = ax_ref.get_position()
    cax.set_position([p.x0, p.y0, w_in/W, h_in/H])
    tb = cax.get_tightbbox(renderer).transformed(to_fig)
    left_in = (p.x0 - tb.x0)*W                 # group extent left of the bar
    top_in = (tb.y1 - (p.y0 + h_in/H))*H       # group extent above the bar

    if state['added_in']:
        _resize_fig_height(fig, -state['added_in'], upper_axes=[ax_ref, cax])
        state['added_in'] = 0.0
        W, H = fig.get_size_inches()

    p = ax_ref.get_position()
    anchor = state['inset_anchor']
    if anchor is None:
        anchor = (0.03, 0.97) if state.get('style_used') == 'cropped' else (0.01, 0.97)
    ax_x = p.x0 + anchor[0]*p.width
    ax_y = p.y0 + anchor[1]*p.height
    x0 = ax_x + left_in/W
    y1 = ax_y - top_in/H
    cax.set_position([x0, y1 - h_in/H, w_in/W, h_in/H])


def _place_shared_tbar_between(fig, state, ax_ref):
    '''Position the bar (with brackets/labels) just below ax_ref and resize the figure so the
    gap between the rows fits it exactly.

    All extents are measured once at the current figure size. They are rigid in inches (labels,
    tick labels, and axes keep their inch sizes under _resize_fig_height), so a single resize is
    exact. (Re-measuring text right after a resize can return stale extents.)'''
    cax = state['ax']
    renderer = fig.canvas.get_renderer()
    to_fig = fig.transFigure.inverted()
    W, H = fig.get_size_inches()

    if state['ax_col'] is not None:
        p = state['ax_col'].get_position()
        x0, x1 = p.x0, p.x1
    else:
        lo, hi = cax.get_xlim()
        x0 = (ax_ref.transData + to_fig).transform((lo, 0))[0]
        x1 = (ax_ref.transData + to_fig).transform((hi, 0))[0]
    h = state['row_height']*len(state['rows'])/H
    cax.set_position([x0, ax_ref.get_position().y0 - h, x1 - x0, h])

    # extents of the whole bar group (brackets, labels, tick labels) relative to the bar itself
    tb = cax.get_tightbbox(renderer).transformed(to_fig)
    above = tb.y1 - (ax_ref.get_position().y0)
    below = (ax_ref.get_position().y0 - h) - tb.y0

    # highest allowed top: bottom of ax_ref, or of its x-label where they overlap horizontally
    ceiling = ax_ref.get_position().y0
    lab = ax_ref.xaxis.label
    if lab.get_text() and lab.get_visible():
        lb = lab.get_window_extent(renderer).transformed(to_fig)
        if lb.x1 > tb.x0 and lb.x0 < tb.x1:
            ceiling = min(ceiling, lb.y0)
    bar_top = ceiling - state['pad_top']/H - above
    cax.set_position([x0, bar_top - h, x1 - x0, h])
    group_bottom = bar_top - h - below

    # lower panels under the bar group (horizontally overlapping)
    lower = [a for a in fig.axes if a is not ax_ref and a is not cax]
    lbs = [a.get_tightbbox(renderer).transformed(to_fig) for a in lower]
    under = [b for b in lbs if b.x1 > tb.x0 and b.x0 < tb.x1] or lbs
    lower_top = max(b.y1 for b in under)
    delta_in = (lower_top - group_bottom)*H + state['pad_bottom']
    _resize_fig_height(fig, delta_in, upper_axes=[ax_ref, cax])
    state['added_in'] += delta_in

##### f-gamma timescans #####
def plot_fg_man(df: pd.DataFrame | rd.ResultCollection, modes, fs, gs, ax, legend=True, **ckws):

    if isinstance(df, pd.DataFrame):
        runs = df['run'].unique()
    else:
        coll = df
        runs = list(coll.idx.keys())
    palette_names = ckws.pop('palette', ['Blues', 'Greens', 'Oranges', 'Purples', 'Reds'])
    palettes = [sns.color_palette(palette, n_colors=len(runs)+1) for palette in palette_names]

    for mode in fs.keys():
        ax.scatter(x=fs[mode], y=gs[mode], color='k', marker='*', s=300, lw=0, zorder=10)
        ax.text(s=f'{mode[0]}{mode[1]}{mode[2]}', x=fs[mode], y=gs[mode]-(0.075*(max(gs.values()) - min(gs.values()))), ha='center', va='top')

    for i, run in enumerate(runs[::-1]):
        for m, mode in enumerate(modes):
            lw = 2
            if run == 0:
                lw = 4
            if isinstance(df, pd.DataFrame):
                sns.kdeplot(df[df['run'] == run], x=f'f_{mode}', y=f'g_{mode}', 
                            color=palettes[m][i], alpha=(i+1)/(len(runs)),
                            linewidths=lw, levels=[0.1], ax=ax)
            else:
                sns.kdeplot(x=coll.idx[run].posterior.f[:,:,mode].values.flatten()[::4],
                            y=coll.idx[run].posterior.g[:,:,mode].values.flatten()[::4], 
                            color=palettes[m][i], alpha=(i+1)/(len(runs)),
                            linewidths=lw, levels=[0.1], ax=ax)

    ax.set_xlabel('$f$ [Hz]')
    ax.set_ylabel('$\gamma$ [Hz]')

    frange = max(fs.values()) - min(fs.values())
    grange = max(gs.values()) - min(gs.values())
    ax.set_xlim(min(fs.values())-frange*0.25, max(fs.values())+frange*0.25)
    ax.set_ylim(0, max(gs.values())+grange*0.25)

##### mode amplitude evolution #####
def get_projection(df, mode, t0ref):
    t = np.arange(-15, 28, 3)

    TM = remnant_ds.m * rd.qnms.T_MSUN
    tref = (t - t0ref) * TM

    ref_df = df[df['run'] == t0ref]
    a = ref_df[f'a_{mode}'].values[:,None] * np.exp(-ref_df[f'g_{mode}'].values[:,None] * tref[None,:])
    a_df = pd.DataFrame(a, columns=t)

    return a_df

def plot_projection(a_df, cis, ax, color='gray', a_scale=1e-21):
    t = np.array(a_df.columns.values)
    for ci in cis:
        lohi = a_df.apply(lambda x: az.hdi(x.values, ci)).T
        ax.fill_between(
            t, 
            lohi.iloc[:,0]/a_scale, lohi.iloc[:,1]/a_scale,
            alpha=0.1, 
            color=color, 
            zorder=-100,
            lw=0
        )

def plot_scan(df, mode, cis, color, ax, a_scale=1e-21, marker=None, alpha_scale=1, shift=0):
    kws = {}

    m = df.groupby('run')[f'a_{mode}'].apply(lambda x: x.median())
    for i, ci in enumerate(cis):
        lohi = df.groupby('run')[f'a_{mode}'].apply(lambda x: pd.Series(az.hdi(x.values, ci))).unstack(level=1)
        
        if i == 0:
            kws['capsize'] = 0
            kws['lw'] = 2
        else:
            kws['capsize'] = None
            kws['lw'] = 5

        _, caps, bars = ax.errorbar(m.index+shift, m/a_scale, ((m - lohi.iloc[:,0])/a_scale, (lohi.iloc[:,1] - m)/a_scale), marker=" ", markersize=1, linestyle="", c=color, alpha=0.5*alpha_scale, **kws)
        for bar in bars:
            bar.set_capstyle('round')
        if marker is not None:
            ax.plot(m.index +shift, m/a_scale, marker=marker,  markersize=7, color=color, alpha=0.95*alpha_scale, linestyle="")
