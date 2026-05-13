# chrom_justification.py

import os
import math
import time
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.gridspec import GridSpec

from core.signal_proc import SignalPreprocessor, SGFilter
from core.rppg import rPPGProcessor

try:
    from sklearn.decomposition import FastICA
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

CHROM_JUSTIFY_CONFIG = {
    'mode': 'C_combined',   
    'fps': 30.0,
    'bpm_min': 45.0,
    'bpm_max': 150.0,
    'output_dir': 'rppg_output',  
    'fig_dpi': 110,
    'savefig': True,
}

def _get_raw_signals(cfg, real_data=None):
    if real_data is not None:
        R = np.array(real_data.get('R',[]), dtype=np.float64)
        G = np.array(real_data.get('G',[]), dtype=np.float64)
        B = np.array(real_data.get('B',[]), dtype=np.float64)
        N = min(len(R), len(G), len(B))
        if N >= 60:
            return R[:N], G[:N], B[:N], False

    fps = cfg['fps']
    N = int(12 * fps)
    t = np.arange(N) / fps
    hr = 1.2
    resp = 0.25
    noise = np.random.default_rng(42)

    G = 100 + 8 * np.sin(2 * np.pi * hr * t) + 2 * np.sin(2 * np.pi * 2 * hr * t) + 3 * np.sin(2 * np.pi * resp * t) + noise.normal(0, 1.5, N)
    R = 120 + 5 * np.sin(2 * np.pi * hr * t + 0.3) + 4 * np.sin(2 * np.pi * resp * t) + noise.normal(0, 2.5, N)
    B = 80 + 2 * np.sin(2 * np.pi * hr * t + 0.6) + 5 * np.sin(2 * np.pi * resp * t) + noise.normal(0, 3.0, N) + 0.05 * t
    return R, G, B, True

def _pipeline(sig, preproc, sg):
    bpf = preproc.filter(sig.copy())
    out = sg.smooth(bpf)
    p = np.max(np.abs(out))
    return out / p if p > 1e-10 else out

def _fft_analysis(sig, fps, bpm_min=45.0, bpm_max=150.0):
    N = len(sig)
    win = np.hanning(N)
    sig_w = (sig - sig.mean()) * win
    Nf = 1
    while Nf < N: Nf <<= 1
    Nf *= 2    
    X = np.fft.rfft(sig_w, n=Nf)
    mags = np.abs(X)
    freqs = np.fft.rfftfreq(Nf, d=1.0/fps)
    f_lo, f_hi = bpm_min / 60.0, bpm_max / 60.0
    band = (freqs >= f_lo) & (freqs <= f_hi)
    noise = ~band & (freqs > 0.1) & (freqs < fps/2)
    sig_pwr = float(np.sum(mags[band]**2)) + 1e-12
    noise_pwr = float(np.sum(mags[noise]**2)) + 1e-12
    snr_db = 10 * np.log10((sig_pwr / max(band.sum(), 1)) / (noise_pwr / max(noise.sum(), 1)))
    pi = np.argmax(mags[band])
    return freqs, mags, float(snr_db), float(freqs[band][pi]) * 60.0, float(freqs[band][pi])

def _chrom_decompose(R, G, B):
    eps = 1e-8
    lum = R + G + B + eps
    Rn, Gn, Bn = R / lum, G / lum, B / lum
    Xs = 3 * Rn - 2 * Gn
    Ys = 1.5 * Rn + Gn - 1.5 * Bn
    alpha = np.std(Xs) / (np.std(Ys) + eps)
    S = Xs - alpha * Ys
    def _n(x): p = np.max(np.abs(x)); return x / p if p > 1e-10 else x
    return _n(Xs), _n(Ys), _n(S), float(alpha)

def _ica_best_component(R, G, B, fps, bpm_min, bpm_max):
    X = np.column_stack([R - R.mean(), G - G.mean(), B - B.mean()])
    if _HAS_SKLEARN:
        try:
            S = FastICA(n_components=3, random_state=0, max_iter=500, tol=0.01).fit_transform(X)      
        except Exception:
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            S = X @ Vt.T
    else:
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        S = X @ Vt.T

    best_comp, best_snr = S[:, 0], -999.0
    for i in range(S.shape[1]):
        comp = S[:, i]
        p = np.max(np.abs(comp))
        comp_n = comp / p if p > 1e-10 else comp
        _, _, snr, _, _ = _fft_analysis(comp_n, fps, bpm_min, bpm_max)
        if snr > best_snr:
            best_snr, best_comp = snr, comp_n
    return best_comp

def run_chrom_justification(cfg=None, real_data=None):
    config = CHROM_JUSTIFY_CONFIG.copy()
    if cfg is not None:
        config.update(cfg)

    fps, bpm_min, bpm_max = config['fps'], config['bpm_min'], config['bpm_max']

    R, G, B, is_synthetic = _get_raw_signals(config, real_data)
    N = len(R)
    t = np.arange(N) / fps

    preproc = SignalPreprocessor(fps=fps, low_hz=bpm_min/60, high_hz=bpm_max/60)
    sg = SGFilter(window_len=11, poly_order=3)

    ch_sigs = {ch: _pipeline(raw.copy(), preproc, sg) for ch, raw in {'R': R, 'G': G, 'B': B}.items()}

    R_sg = _pipeline(R.copy(), preproc, sg) * np.std(R)   
    G_sg = _pipeline(G.copy(), preproc, sg) * np.std(G)
    B_sg = _pipeline(B.copy(), preproc, sg) * np.std(B)
    
    Xs, Ys, S_raw, alpha = _chrom_decompose(R, G, B)
    rppg_proc = rPPGProcessor(method='chrom')
    chrom_raw = rppg_proc.compute({'R': R_sg + R.mean(), 'G': G_sg + G.mean(), 'B': B_sg + B.mean()})
    chrom_sig = _pipeline(chrom_raw, preproc, sg)

    ica_sig = _pipeline(_ica_best_component(R, G, B, fps, bpm_min, bpm_max), preproc, sg)

    methods = {'CHROM': chrom_sig, 'Green': ch_sigs['G'], 'Red': ch_sigs['R'], 'Blue': ch_sigs['B'], 'ICA': ica_sig}
    
    fft_results = {}
    for name, sig in methods.items():
        freqs, mags, snr, bpm, hz = _fft_analysis(sig, fps, bpm_min, bpm_max)
        fft_results[name] = dict(freqs=freqs, mags=mags, snr=snr, bpm=bpm, hz=hz, sig=sig)

    # Acquire DPI value from config and set up figure 
    dpi = config['fig_dpi']
    fig = Figure(figsize=(20, 14), dpi=dpi, facecolor='#10101a')
    canvas = FigureCanvasAgg(fig)

    PALETTE = {'CHROM': '#00e5ff', 'Green': '#00dd44', 'Red': '#ff4444', 'Blue': '#4488ff', 'ICA': '#ffcc00'}
    BG, PANEL, GRID_C, TEXT_C, ACCENT = '#10101a', '#181828', '#2a2a3a', '#ccccdd', '#7755ee'

    gs = GridSpec(3, 4, figure=fig, left=0.05, right=0.97, top=0.93, bottom=0.07, hspace=0.55, wspace=0.35)

    src_label = '(Synthetic Signal)' if is_synthetic else f'(Real-time Data — Mode {config.get("mode", "C_combined")})'
    fig.text(0.5, 0.97, f'Comparative Analysis of rPPG Methods: CHROM vs Single-Channel vs ICA  {src_label}', ha='center', va='top', fontsize=13, fontweight='bold', color='#e0e0ff', family='monospace')

    # PANEL A: Waveform
    wave_order_display =['CHROM', 'Green', 'Red', 'Blue']
    for ci, name in enumerate(wave_order_display):
        ax = fig.add_subplot(gs[0, ci])
        sig = methods[name]
        t_plot = t[:len(sig)]
        ax.plot(t_plot, sig, color=PALETTE[name], linewidth=1.0, alpha=0.9)
        ax.set_facecolor(PANEL)
        ax.set_title(f'{name} — {fft_results[name]["bpm"]:.1f} BPM | SNR {fft_results[name]["snr"]:+.1f} dB', color=PALETTE[name], fontsize=8.5, pad=4, fontweight='bold')
        ax.set_xlabel('Time (seconds)', color=TEXT_C, fontsize=7.5)
        ax.set_ylabel('Amplitude (normalized)', color=TEXT_C, fontsize=7.5)
        ax.tick_params(colors='#777788', labelsize=7, labelcolor=TEXT_C)
        ax.spines[:].set_color(GRID_C)
        ax.grid(True, color=GRID_C, linewidth=0.4, linestyle='--', alpha=0.7)
        ax.axhline(0, color='#333344', linewidth=0.6)
        ax.set_xlim(t_plot[0], t_plot[-1])

    # PANEL B: FFT Spectrum
    for ci, name in enumerate(wave_order_display):
        ax = fig.add_subplot(gs[1, ci])
        fr = fft_results[name]
        freqs_hz, mags = fr['freqs'], fr['mags']
        mags_n = mags / (mags.max() + 1e-12)
        band = (freqs_hz >= bpm_min/60) & (freqs_hz <= bpm_max/60)
        
        ax.plot(freqs_hz, mags_n, color='#444455', linewidth=0.8, alpha=0.6)
        ax.fill_between(freqs_hz, mags_n, where=band, color=PALETTE[name], alpha=0.35)
        ax.plot(freqs_hz[band], mags_n[band], color=PALETTE[name], linewidth=1.4)
        
        peak_hz = fr['hz']
        peak_idx = np.argmin(np.abs(freqs_hz - peak_hz))
        ax.axvline(peak_hz, color='#ffffff', linewidth=1.0, linestyle='--', alpha=0.7)
        ax.scatter([peak_hz], [mags_n[peak_idx]], color='white', s=40, zorder=5)
        ax.annotate(f"{fr['bpm']:.1f} BPM", xy=(peak_hz, mags_n[peak_idx]), xytext=(peak_hz + 0.08, mags_n[peak_idx] * 0.85), color='white', fontsize=7, fontweight='bold', arrowprops=dict(arrowstyle='->', color='white', lw=0.8))
        
        ax.axvspan(bpm_min/60, bpm_max/60, alpha=0.06, color=PALETTE[name])
        ax.set_facecolor(PANEL)
        ax.set_title(f'FFT — {name}', color=PALETTE[name], fontsize=8.5, pad=4, fontweight='bold')
        ax.set_xlabel('Frequency (Hz)', color=TEXT_C, fontsize=7.5)
        ax.set_ylabel('Magnitude (norm.)', color=TEXT_C, fontsize=7.5)
        ax.tick_params(colors=TEXT_C, labelsize=7)
        ax.spines[:].set_color(GRID_C)
        ax.grid(True, color=GRID_C, linewidth=0.4, linestyle='--', alpha=0.7)
        ax.set_xlim(0.0, bpm_max/60 + 0.3)
        ax.text(0.03, 0.95, f'SNR: {fr["snr"]:+.2f} dB', transform=ax.transAxes, color=PALETTE[name], fontsize=7.5, fontweight='bold', va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='#000000aa', edgecolor=PALETTE[name], linewidth=0.8))

    # PANEL C: CHROM Decomposition (Xs, Ys, S)
    ax_decomp = fig.add_subplot(gs[2, :2])
    ax_decomp.set_facecolor(PANEL)
    N_dec = min(len(Xs), len(Ys), len(S_raw), len(t))
    t_d = t[:N_dec]
    ax_decomp.plot(t_d, Xs[:N_dec], color='#ff9900', linewidth=0.9, alpha=0.85, label=r'$X_s = 3R_n - 2G_n$')
    ax_decomp.plot(t_d, Ys[:N_dec], color='#cc66ff', linewidth=0.9, alpha=0.85, label=r'$Y_s = 1.5R_n + G_n - 1.5B_n$')
    ax_decomp.plot(t_d, S_raw[:N_dec], color=PALETTE['CHROM'], linewidth=1.3, label=rf'$S = X_s - {alpha:.2f} \cdot Y_s$')
    ax_decomp.axhline(0, color='#333344', linewidth=0.5)
    ax_decomp.set_title(r'Chrominance CHROM Decomposition — Elimination of Specular Reflection components', color='#e0e0ff', fontsize=8.5, pad=4, fontweight='bold')
    ax_decomp.set_xlabel('Time (seconds)', color=TEXT_C, fontsize=7.5)
    ax_decomp.set_ylabel('Amplitude (norm.)', color=TEXT_C, fontsize=7.5)
    ax_decomp.tick_params(colors=TEXT_C, labelsize=7)
    ax_decomp.spines[:].set_color(GRID_C)
    ax_decomp.grid(True, color=GRID_C, linewidth=0.4, linestyle='--', alpha=0.7)
    ax_decomp.set_xlim(t_d[0], t_d[-1])
    ax_decomp.legend(fontsize=7, loc='upper right', facecolor='#1a1a2a', labelcolor=TEXT_C, framealpha=0.85, edgecolor=GRID_C)
    ax_decomp.text(0.02, 0.05, rf'$\alpha = \sigma(X_s)/\sigma(Y_s) = {alpha:.3f}$  → membatalkan specular noise', transform=ax_decomp.transAxes, color='#aaaacc', fontsize=7.5, va='bottom', bbox=dict(boxstyle='round,pad=0.3', facecolor='#000000aa', edgecolor=ACCENT, linewidth=0.8))

    # PANEL D: ICA waveform
    ax_ica = fig.add_subplot(gs[2, 2])
    ax_ica.set_facecolor(PANEL)
    t_ica = t[:len(ica_sig)]
    ax_ica.plot(t_ica, ica_sig, color=PALETTE['ICA'], linewidth=0.95, alpha=0.9, label='ICA (best component)')
    ax_ica.axhline(0, color='#333344', linewidth=0.5)
    method_label = 'FastICA' if _HAS_SKLEARN else 'PCA-ICA'
    ax_ica.set_title(f'ICA — {method_label}\n{fft_results["ICA"]["bpm"]:.1f} BPM | SNR {fft_results["ICA"]["snr"]:+.1f} dB', color=PALETTE['ICA'], fontsize=8, pad=4, fontweight='bold')
    ax_ica.set_xlabel('Time (seconds)', color=TEXT_C, fontsize=7.5)
    ax_ica.set_ylabel('Amplitude (normalized)', color=TEXT_C, fontsize=7.5)
    ax_ica.tick_params(colors=TEXT_C, labelsize=7)
    ax_ica.spines[:].set_color(GRID_C)
    ax_ica.grid(True, color=GRID_C, linewidth=0.4, linestyle='--', alpha=0.7)
    ax_ica.set_xlim(t_ica[0], t_ica[-1])
    ax_ica.legend(fontsize=7, loc='upper right', facecolor='#1a1a2a', labelcolor=TEXT_C, framealpha=0.85, edgecolor=GRID_C)

    # PANEL E: SNR Bar Chart
    ax_snr = fig.add_subplot(gs[2, 3])
    ax_snr.set_facecolor(PANEL)
    all_methods =['CHROM', 'ICA', 'Green', 'Red', 'Blue']
    all_snr = [fft_results[m]['snr'] for m in all_methods]
    all_colors = [PALETTE[m] for m in all_methods]
    all_bpm =[fft_results[m]['bpm'] for m in all_methods]

    bars = ax_snr.barh(all_methods, all_snr, color=all_colors, alpha=0.85, edgecolor='#ffffff44', linewidth=0.6, height=0.55)
    for bar, snr_val, bpm_val in zip(bars, all_snr, all_bpm):
        xpos = snr_val + (0.15 if snr_val >= 0 else -0.15)
        ha = 'left' if snr_val >= 0 else 'right'
        ax_snr.text(xpos, bar.get_y() + bar.get_height()/2, f'{snr_val:+.2f} dB\n({bpm_val:.0f} BPM)', va='center', ha=ha, fontsize=6.5, color='white', fontweight='bold')
        
    ax_snr.axvline(0, color='#ffffff55', linewidth=0.8, linestyle='--')
    bars[0].set_edgecolor('#ffffff'); bars[0].set_linewidth(1.5)
    best_idx = int(np.argmax(all_snr))
    ax_snr.text(0.5, 1.01, f'↑ Best: {all_methods[best_idx]} ({all_snr[best_idx]:+.2f} dB)', transform=ax_snr.transAxes, ha='center', va='bottom', fontsize=7.5, color=all_colors[best_idx], fontweight='bold')
    ax_snr.set_title('SNR in Pulse Band (45–150 BPM)', color=TEXT_C, fontsize=8.5, pad=4, fontweight='bold')
    ax_snr.set_xlabel('SNR (dB)', color=TEXT_C, fontsize=7.5)
    ax_snr.tick_params(colors=TEXT_C, labelsize=7.5, labelcolor=TEXT_C)
    ax_snr.spines[:].set_color(GRID_C)
    ax_snr.grid(True, axis='x', color=GRID_C, linewidth=0.4, linestyle='--', alpha=0.7)

    # Render and Save
    canvas.draw()
    
    out_dir = config.get('output_dir', 'rppg_output')
    os.makedirs(out_dir, exist_ok=True)
    
    ts = time.strftime('%Y%m%d_%H%M%S')
    filename = f'chrom_justification_{ts}.png'
    path = os.path.join(out_dir, filename)
    
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=BG)
    
    abs_path = os.path.abspath(path)
    print(f"\n[CHROM-Justify] ✔ File automatically saved in:\n  -> {abs_path}\n")

    return abs_path, fft_results

if __name__ == "__main__":
    run_chrom_justification()