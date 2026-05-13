# monitor/data_saver.py

import os
import time
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

class MultiROIDataSaver:
    MODE_LABELS = {
        'A_forehead': 'Mode A - Forehead',
        'B_cheek': 'Mode B - Cheek',
        'C_combined': 'Mode C - Combined'
    }
    
    MODE_COLORS = {
        'A_forehead': '#ff6644', 
        'B_cheek': '#44aaff', 
        'C_combined': '#55ee55'
    }
    
    CH_STYLE = {
        'R': ('#ff9988', '#ff2222'), 
        'G': ('#88dd99', '#00cc33'), 
        'B': ('#88aaff', '#2244ff')
    }

    def __init__(self, output_dir='rppg_output'):
        self.output_dir = output_dir

    def save_all(self, results, fps, bpm_log=None):
        os.makedirs(self.output_dir, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')

        self._save_csv(results, fps, ts)
        self._plot_rppg_all(results, fps, ts)
        self._plot_fft_overlay(results, ts)
        self._plot_bpf_rgb(results, fps, ts)
        self._plot_hr_comparison(results, ts)
        self._plot_sg_effect(results, fps, ts)
        self._plot_channel_fft(results, fps, ts)
        
        if bpm_log:
            self._plot_kalman_history(bpm_log, ts)

        print(f"[MultiROIDataSaver] All files saved to: '{self.output_dir}/'")
        return self.output_dir

    def _save_csv(self, results, fps, ts):
        path = os.path.join(self.output_dir, f'rppg_data_{ts}.csv')
        Ns = {}
        for m, d in results.items():
            rs = d.get('raw_signals', {})
            Ns[m] = min(
                len(rs.get('R', [])), len(rs.get('G',[])), len(rs.get('B', [])),
                len(d.get('rppg_norm',[])), len(d.get('filt_r', [])),
                len(d.get('filt_g',[])), len(d.get('filt_b',[]))
            )
            
        N_max = max(Ns.values()) if Ns else 0
        if N_max == 0:
            print("  CSV -> skip")
            return
            
        with open(path, 'w') as f:
            cols =['frame_idx', 'timestamp']
            for m in ('A_forehead', 'B_cheek', 'C_combined'):
                for ch in['R_raw', 'G_raw', 'B_raw', 'R_BPF', 'G_BPF', 'B_BPF', 'R_BPFSG', 'G_BPFSG', 'B_BPFSG', 'rPPG']:
                    cols.append(f'{m}_{ch}')
            f.write(','.join(cols) + '\n')
            
            for i in range(N_max):
                t_arr = results.get('A_forehead', {}).get('raw_signals', {}).get('time', np.array([]))
                row =[str(i), f'{t_arr[i]:.6f}' if i < len(t_arr) else '0']
                
                for m in ('A_forehead', 'B_cheek', 'C_combined'):
                    d = results.get(m, {})
                    rs = d.get('raw_signals', {})
                    v = lambda a: f'{a[i]:.6f}' if i < len(a) else ''
                    
                    row += [
                        v(rs.get('R', [])), v(rs.get('G', [])), v(rs.get('B',[])),
                        v(d.get('bpf_r', [])), v(d.get('bpf_g',[])), v(d.get('bpf_b',[])),
                        v(d.get('filt_r', [])), v(d.get('filt_g',[])), v(d.get('filt_b', [])),
                        v(d.get('rppg_norm',[]))
                    ]
                f.write(','.join(row) + '\n')
        print(f"  CSV         -> {path}")

    def _plot_rppg_all(self, results, fps, ts):
        modes = ['A_forehead', 'B_cheek', 'C_combined']
        fig = Figure(figsize=(16, 10), dpi=120, facecolor='#1a1a2e')
        canvas = FigureCanvasAgg(fig)
        
        for ri, mode in enumerate(modes):
            d = results.get(mode, {})
            rppg = d.get('rppg_norm', np.array([]))
            if len(rppg) < 4:
                continue
                
            N = len(rppg)
            t = np.arange(N, dtype=np.float64) / fps
            ax = fig.add_subplot(3, 1, ri + 1)
            ax.set_facecolor('#0d1117')
            c = self.MODE_COLORS[mode]
            
            ax.plot(t, rppg, color=c, linewidth=1.0, label='rPPG (CHROM+SG)')
            ax.axhline(0, color='#444455', linewidth=0.6, linestyle='--')
            
            pr = d.get('peak_result', {})
            pks = np.array(pr.get('peak_locs', []))
            pks = pks[pks < N]
            if len(pks) > 0:
                ax.scatter(t[pks], rppg[pks], color='#ffee22', s=50, zorder=5, marker='v', label=f'{len(pks)} peaks')
                for j in range(min(len(pks) - 1, 10)):
                    mid_t = (t[pks[j]] + t[pks[j+1]]) / 2
                    ibi_ms = (t[pks[j+1]] - t[pks[j]]) * 1000
                    ax.annotate(f'{ibi_ms:.0f}ms', xy=(mid_t, rppg[pks[j]] + .05),
                                fontsize=7.5, color='#ffaa33', ha='center')
                                
            ff = d.get('fft_result', {})
            bk = ff.get('bpm_kalman')
            bm = ff.get('bpm_median')
            conf = ff.get('confidence', 0)
            pbpm = pr.get('bpm')
            
            title = self.MODE_LABELS[mode]
            if bk: title += f' | Kalman = {bk:.1f} BPM'
            if bm: title += f' | Median = {bm:.1f}'
            if pbpm: title += f' | Peak-domain = {pbpm:.1f}'
            if conf: title += f' | conf = {conf*100:.1f}%'
            
            ax.set_title(title, color=c, fontsize=11, pad=5, fontweight='bold')
            ax.set_ylabel('Amplitude (normalized)', color='#cccccc', fontsize=10)
            ax.tick_params(colors='#888888')
            for sp in ax.spines.values():
                sp.set_edgecolor('#444')
                
            ax.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white', framealpha=0.8)
            ax.set_xlim(t[0], t[-1])
            if ri == 2:
                ax.set_xlabel('Time (seconds)', color='#cccccc', fontsize=10)
                
        fig.suptitle('rPPG Signal + Peak Detection - Mode ROI A/B/C', color='white', fontsize=13, y=0.98, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        path = os.path.join(self.output_dir, f'plot_rppg_all_{ts}.png')
        canvas.print_figure(path, dpi=120)
        print(f"  rPPG        -> {path}")

    def _plot_fft_overlay(self, results, ts):
        fig = Figure(figsize=(14, 5), dpi=120, facecolor='#1a1a2e')
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.set_facecolor('#0d1117')
        
        ax.axvspan(0.75, 2.5, alpha=0.08, color='#00ff88', label='HR valid zone (45-150 BPM)')
        for br, lb in[(60, '60'), (75, '75'), (90, '90'), (120, '120'), (150, '150')]:
            ax.axvline(br/60, color='#444455', linewidth=0.6, linestyle=':', alpha=0.7)
            ax.text(br/60, 0, lb, color='#666677', fontsize=8, ha='center', va='bottom')
            
        ax.axvline(2.5, color='#ff4444', linewidth=1.0, linestyle='--', alpha=0.7, label='BPF cutoff (2.5 Hz = 150 BPM)')
        
        ann =[]
        for mode in ('A_forehead', 'B_cheek', 'C_combined'):
            d = results.get(mode, {})
            ff = d.get('fft_result', {})
            freqs = ff.get('frequencies', np.array([]))
            mags = ff.get('magnitudes', np.array([]))
            
            if len(freqs) == 0:
                continue
                
            c = self.MODE_COLORS[mode]
            mask = freqs <= 4.0
            fv, mv = freqs[mask], mags[mask]
            mv_n = mv / (mv.max() + 1e-8)
            
            ax.plot(fv, mv_n, color=c, linewidth=1.2, alpha=0.9, label=self.MODE_LABELS[mode])
            ax.fill_between(fv, 0, mv_n, alpha=0.12, color=c)
            
            pf = ff.get('peak_freq')
            bpm = ff.get('bpm')
            conf = ff.get('confidence', 0)
            
            if pf:
                pfn = float(mags[np.argmin(np.abs(freqs - pf))]) / (mags.max() + 1e-8)
                ax.axvline(pf, color=c, linewidth=1.5, linestyle='--', alpha=0.8)
                ax.scatter([pf], [pfn], color=c, s=70, zorder=6)
                ann.append(f"{self.MODE_LABELS[mode].split('-')[1].strip()}: {bpm:.1f} BPM (conf {conf*100:.1f}%)")
                
        ax.set_title('FFT Spectrum Overlay: Mode A vs B vs C\n' + '  |  '.join(ann), color='white', fontsize=11, pad=6)
        ax.set_xlabel('Frequency (Hz)', color='#cccccc', fontsize=10)
        ax.set_ylabel('|X[k]| (norm.)', color='#cccccc', fontsize=10)
        ax.tick_params(colors='#888888')
        for sp in ax.spines.values():
            sp.set_edgecolor('#444')
            
        ax.set_xlim(0, 4.0)
        ax.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white', framealpha=0.9, loc='upper right')
        fig.tight_layout()
        
        path = os.path.join(self.output_dir, f'plot_fft_overlay_{ts}.png')
        canvas.print_figure(path, dpi=120)
        print(f"  FFT overlay -> {path}")

    def _plot_bpf_rgb(self, results, fps, ts):
        modes =['A_forehead', 'B_cheek', 'C_combined']
        fig = Figure(figsize=(16, 10), dpi=110, facecolor='#1a1a2e')
        canvas = FigureCanvasAgg(fig)
        
        for col, mode in enumerate(modes):
            d = results.get(mode, {})
            rs = d.get('raw_signals', {})
            N_l = min(len(rs.get('R', [])), len(rs.get('G', [])), len(rs.get('B',[])),
                      len(d.get('filt_r', [])), len(d.get('filt_g',[])), len(d.get('filt_b',[])))
                      
            if N_l < 4:
                continue
                
            t = np.arange(N_l, dtype=np.float64) / fps
            for row, (ch, rk, fk) in enumerate([('R','R','filt_r'), ('G','G','filt_g'), ('B','B','filt_b')]):
                ax = fig.add_subplot(3, 3, row * 3 + col + 1)
                ax.set_facecolor('#0d1117')
                rc, fc = self.CH_STYLE[ch]
                
                raw_a = rs.get(rk, np.array([]))[:N_l]
                raw_dc = raw_a - raw_a.mean()
                filt_a = d.get(fk, np.array([]))[:N_l]
                
                ax.plot(t, raw_dc, color=rc, linewidth=0.8, alpha=0.45, label=f'{ch} raw (DC-rem)')
                ax.plot(t, filt_a, color=fc, linewidth=1.0, label=f'{ch} BPF+SG')
                ax.axhline(0, color='#333344', linewidth=0.4, linestyle='--')
                
                mc = self.MODE_COLORS[mode]
                ax.set_title(f'[{mode.split("_")[1].capitalize()}] Ch {ch}', color=mc, fontsize=10, pad=3)
                ax.tick_params(colors='#888888', labelsize=9)
                for sp in ax.spines.values():
                    sp.set_edgecolor('#444')
                    
                ax.set_xlim(t[0], t[-1])
                ax.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white', framealpha=0.7, loc='upper right')
                
                if col == 0:
                    ax.set_ylabel(f'Ch {ch} (amp.)', color='#aaa', fontsize=9)
                if row == 2:
                    ax.set_xlabel('Time (s)', color='#aaa', fontsize=9)
                    
        fig.suptitle('BPF IIR + SG Smoothing, Raw vs Filtered: R/G/B x Mode A/B/C', color='white', fontsize=12, y=0.98, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        path = os.path.join(self.output_dir, f'plot_bpf_rgb_{ts}.png')
        canvas.print_figure(path, dpi=110)
        print(f"  BPF RGB     -> {path}")

    def _plot_hr_comparison(self, results, ts):
        fig = Figure(figsize=(15, 5.5), dpi=120, facecolor='#1a1a2e')
        canvas = FigureCanvasAgg(fig)
        ax_bar = fig.add_subplot(1, 2, 1)
        ax_tbl = fig.add_subplot(1, 2, 2)
        ax_bar.set_facecolor('#0d1117')
        ax_tbl.set_facecolor('#0d1117')
        ax_tbl.axis('off')
        
        modes = ['A_forehead', 'B_cheek', 'C_combined']
        labels =['Forehead (A)', 'Cheek (B)', 'Combined (C)']
        colors =[self.MODE_COLORS[m] for m in modes]
        bk_l, bm_l, br_l, bp_l, conf_l, trend_l = [], [], [], [], [],[]
        
        for mode in modes:
            d = results.get(mode, {})
            ff = d.get('fft_result', {})
            pr = d.get('peak_result', {})
            
            bk_l.append(ff.get('bpm_kalman') or 0)
            bm_l.append(ff.get('bpm_median') or 0)
            br_l.append(ff.get('bpm') or 0)
            bp_l.append(pr.get('bpm') or 0)
            conf_l.append(ff.get('confidence', 0) * 100)
            trend_l.append(ff.get('kalman_trend', 0.0))
            
        x = np.arange(len(modes))
        w = 0.2
        
        for xi, (vals, lbl, alpha) in enumerate([
                (bk_l, 'BPM Kalman', 1.0),
                (bm_l, 'BPM Median', 0.75),
                (br_l, 'BPM Raw FFT', 0.5),
                (bp_l, 'BPM Peak-dom', 0.0)]):
                
            offset = (xi - 1.5) * w
            if alpha > 0:
                bars = ax_bar.bar(x + offset, vals, w, label=lbl, color=colors, edgecolor='white', linewidth=0.6, alpha=alpha)
            else:
                bars = ax_bar.bar(x + offset, vals, w, label=lbl, color=['#ffffff']*3, edgecolor=colors, linewidth=1.5, fill=False)
                
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                                f'{v:.1f}', ha='center', va='bottom', color='white', fontsize=8)
                                
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(labels, color='white', fontsize=9)
        ax_bar.set_ylabel('Heart Rate (BPM)', color='#ccc', fontsize=10)
        ax_bar.set_ylim(0, max(max(bk_l + bm_l + br_l + bp_l, default=120) + 25, 50))
        ax_bar.set_title('HR Comparison - Kalman vs Median vs Raw vs Peak', color='white', fontsize=11, pad=6, fontweight='bold')
        ax_bar.tick_params(colors='#888')
        for sp in ax_bar.spines.values():
            sp.set_edgecolor('#444')
            
        ax_bar.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white', framealpha=0.85)
        ax_bar.axhline(60, color='#333', linewidth=0.5, linestyle=':')
        ax_bar.axhline(100, color='#333', linewidth=0.5, linestyle=':')
        
        tbl_data =[]
        for i, mode in enumerate(modes):
            ts_sym = '^' if trend_l[i] > 0.3 else 'v' if trend_l[i] < -0.3 else '->'
            tbl_data.append([
                labels[i],
                f"{bk_l[i]:.1f}" if bk_l[i] else '-',
                f"{bm_l[i]:.1f}" if bm_l[i] else '-',
                f"{br_l[i]:.1f}" if br_l[i] else '-',
                f"{bp_l[i]:.1f}" if bp_l[i] else '-',
                f"{conf_l[i]:.1f}%",
                f"{trend_l[i]:+.2f} {ts_sym}"
            ])
            
        tbl = ax_tbl.table(cellText=tbl_data,
                           colLabels=['Mode', 'Kalman', 'Median', 'Raw', 'Peak', 'Conf', 'Trend'],
                           loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.05, 2.0)
        
        for j in range(7):
            tbl[(0, j)].set_facecolor('#2a2a4a')
            tbl[(0, j)].set_text_props(color='white', fontweight='bold')
            
        for i, mode in enumerate(modes):
            for j in range(7):
                tbl[(i + 1, j)].set_facecolor('#1a1a2e')
                tbl[(i + 1, j)].set_text_props(color=self.MODE_COLORS[mode])
                
        ax_tbl.set_title('Summary of HR Estimates (BPF+SG+Kalman)', color='white', fontsize=10, pad=10, fontweight='bold')
        fig.suptitle('Heart Rate Comparison - Forehead vs Cheek vs Combined', color='white', fontsize=12, y=0.98, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        path = os.path.join(self.output_dir, f'plot_hr_comparison_{ts}.png')
        canvas.print_figure(path, dpi=120)
        print(f"  HR compare  -> {path}")

    def _plot_sg_effect(self, results, fps, ts):
        modes =['A_forehead', 'B_cheek', 'C_combined']
        row_labels =['Raw G (DC-removed)', 'After BPF IIR (0.75-2.5 Hz)', 'After BPF+SG']
        row_colors = ['#ffaa44', '#44aaff', '#44ff88']

        fig = Figure(figsize=(16, 10), dpi=110, facecolor='#1a1a2e')
        canvas = FigureCanvasAgg(fig)

        for col, mode in enumerate(modes):
            d = results.get(mode, {})
            rs = d.get('raw_signals', {})
            raw_g = rs.get('G', np.array([]))
            bpf_g = d.get('bpf_g', np.array([]))
            filt_g = d.get('filt_g', np.array([]))

            N = min(len(raw_g), len(bpf_g), len(filt_g))
            if N < 4:
                continue
                
            t = np.arange(N, dtype=np.float64) / fps
            raw_dc = raw_g[:N] - raw_g[:N].mean()
            bpf_n = bpf_g[:N]
            sg_n = filt_g[:N]

            stages =[raw_dc, bpf_n, sg_n]
            for row, (sig, label, color) in enumerate(zip(stages, row_labels, row_colors)):
                ax = fig.add_subplot(3, 3, row * 3 + col + 1)
                ax.set_facecolor('#0d1117')
                ax.plot(t, sig, color=color, linewidth=0.9, label=label)
                ax.axhline(0, color='#333344', linewidth=0.4, linestyle='--')

                mc = self.MODE_COLORS[mode]
                ax.set_title(f'[{self.MODE_LABELS[mode].split("-")[1].strip()}] {label}', color=mc, fontsize=10, pad=3)
                ax.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white', framealpha=0.7, loc='upper right')
                ax.tick_params(colors='#888888', labelsize=9)
                for sp in ax.spines.values():
                    sp.set_edgecolor('#444')
                    
                ax.set_xlim(t[0], t[-1])
                if col == 0:
                    ax.set_ylabel('Amplitude (a.u.)', color='#aaa', fontsize=9)
                if row == 2:
                    ax.set_xlabel('Time (s)', color='#aaa', fontsize=9)

                ax.text(0.98, 0.97, f'std={sig.std():.4f}', transform=ax.transAxes,
                        color='#ffcc44', fontsize=8, ha='right', va='top')

        fig.suptitle('Savitzky-Golay Smoothing Effect: Stage F-2/F-3 | Green Channel', color='white', fontsize=12, y=0.98, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        path = os.path.join(self.output_dir, f'plot_sg_effect_{ts}.png')
        canvas.print_figure(path, dpi=110)
        print(f"  SG effect   -> {path}")

    def _plot_channel_fft(self, results, fps, ts):
        modes = ['A_forehead', 'B_cheek', 'C_combined']
        ch_map = {'R': 'filt_r', 'G': 'filt_g', 'B': 'filt_b'}

        fig = Figure(figsize=(16, 11), dpi=110, facecolor='#1a1a2e')
        canvas = FigureCanvasAgg(fig)

        for col, mode in enumerate(modes):
            d = results.get(mode, {})
            for row, ch in enumerate(['R', 'G', 'B']):
                ax = fig.add_subplot(3, 3, row * 3 + col + 1)
                ax.set_facecolor('#0d1117')
                rc, fc = self.CH_STYLE[ch]

                sig = d.get(ch_map[ch], np.array([]))
                if len(sig) < 8:
                    ax.set_title(f'[{ch}] wait', color='#888', fontsize=9)
                    continue

                sig_dc = sig - sig.mean()
                win = np.hanning(len(sig_dc))
                X = np.abs(np.fft.rfft(sig_dc * win))
                freqs = np.fft.rfftfreq(len(sig_dc), d=1.0 / fps)

                mask = freqs <= 4.0
                fv, mv = freqs[mask], X[mask]
                mv_n = mv / (mv.max() + 1e-8)

                ax.plot(fv, mv_n, color=fc, linewidth=1.0, label=f'Ch {ch} FFT')
                ax.fill_between(fv, 0, mv_n, alpha=0.2, color=fc)
                ax.axvspan(0.75, 2.5, alpha=0.07, color='#ffffff')
                ax.axvline(2.5, color='#ff4444', linewidth=0.8, linestyle='--', alpha=0.6)

                vz = (freqs >= 0.75) & (freqs <= 2.5)
                if vz.any():
                    mz, fz = X[vz], freqs[vz]
                    pi = np.argmax(mz)
                    pf = fz[pi]
                    pbpm = pf * 60
                    conf_ch = float(mz[pi]) / (float(mz.sum()) + 1e-8)
                    pf_n = float(X[vz][pi]) / (X[mask].max() + 1e-8)
                    ax.axvline(pf, color='#ffff44', linewidth=1.2, linestyle='--')
                    ax.scatter([pf],[pf_n], color='#ffff44', s=55, zorder=6)
                    ax.text(0.98, 0.97, f'{pbpm:.1f} BPM ({conf_ch*100:.0f}%)',
                            transform=ax.transAxes, color='#ffff44', fontsize=8, ha='right', va='top')

                for bref in[60, 75, 90, 120]:
                    ax.axvline(bref/60, color='#444455', linewidth=0.5, linestyle=':', alpha=0.7)

                mc = self.MODE_COLORS[mode]
                ax.set_title(f'[{self.MODE_LABELS[mode].split("-")[1].strip()}] Ch {ch}: FFT', color=mc, fontsize=10, pad=3)
                ax.tick_params(colors='#888888', labelsize=9)
                for sp in ax.spines.values():
                    sp.set_edgecolor('#444')
                    
                ax.set_xlim(0, 4.0)
                ax.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white', framealpha=0.7, loc='upper right')
                
                if col == 0:
                    ax.set_ylabel('|X[k]| (norm.)', color='#aaa', fontsize=9)
                if row == 2:
                    ax.set_xlabel('Frequency (Hz)', color='#aaa', fontsize=9)

        fig.suptitle('Comparison of FFT Channels R/G/B per ROI Mode', color='white', fontsize=12, y=0.98, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        path = os.path.join(self.output_dir, f'plot_channel_fft_{ts}.png')
        canvas.print_figure(path, dpi=110)
        print(f"  Channel FFT -> {path}")

    def _plot_kalman_history(self, bpm_log, ts):
        modes =['A_forehead', 'B_cheek', 'C_combined']

        fig = Figure(figsize=(15, 10), dpi=110, facecolor='#1a1a2e')
        canvas = FigureCanvasAgg(fig)

        for ri, mode in enumerate(modes):
            log = bpm_log.get(mode,[])
            if len(log) < 2:
                ax = fig.add_subplot(3, 1, ri + 1)
                ax.set_facecolor('#0d1117')
                ax.set_title(f'{self.MODE_LABELS[mode]} - data is not sufficient', color=self.MODE_COLORS[mode], fontsize=11)
                continue

            ts_arr = np.array([e['timestamp'] for e in log])
            br_arr = np.array([e['bpm_raw'] or 0 for e in log])
            bm_arr = np.array([e['bpm_median'] or 0 for e in log])
            bk_arr = np.array([e['bpm_kalman'] or 0 for e in log])
            cf_arr = np.array([e['confidence'] for e in log])

            t_rel = ts_arr - ts_arr[0]

            ax = fig.add_subplot(3, 1, ri + 1)
            ax.set_facecolor('#0d1117')
            c = self.MODE_COLORS[mode]

            ax.fill_between(t_rel, 45, 45 + cf_arr * 100, alpha=0.08, color=c, label='Confidence x100')

            valid_r = br_arr > 0
            ax.scatter(t_rel[valid_r], br_arr[valid_r], color='#888888', s=20, alpha=0.5, zorder=3, label='BPM Raw FFT')

            valid_m = bm_arr > 0
            if valid_m.any():
                ax.step(t_rel[valid_m], bm_arr[valid_m], color='#ffcc33', linewidth=1.0, alpha=0.7, where='mid', label='BPM Median (IQR)')

            valid_k = bk_arr > 0
            if valid_k.any():
                ax.plot(t_rel[valid_k], bk_arr[valid_k], color=c, linewidth=2.0, zorder=5, label='BPM Kalman (output)')

            for ref, lbl in[(60, '60'), (100, '100'), (120, '120')]:
                ax.axhline(ref, color='#333355', linewidth=0.7, linestyle=':', alpha=0.8)
                ax.text(t_rel[-1] * 1.01, ref, lbl, color='#666677', fontsize=8, va='center')

            title = f'{self.MODE_LABELS[mode]} | {len(log)} measurements | '
            title += f'Final Kalman: {bk_arr[valid_k][-1]:.1f} BPM' if valid_k.any() else self.MODE_LABELS[mode]
            
            ax.set_title(title, color=c, fontsize=11, pad=5, fontweight='bold')
            ax.set_ylabel('Heart Rate (BPM)', color='#cccccc', fontsize=10)
            ax.tick_params(colors='#888888')
            for sp in ax.spines.values():
                sp.set_edgecolor('#444')
                
            ax.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white', framealpha=0.85, loc='upper right')
            ax.set_xlim(0, t_rel[-1])
            ax.set_ylim(40, 160)
            
            if ri == 2:
                ax.set_xlabel('Session Time (seconds)', color='#cccccc', fontsize=10)

        fig.suptitle('Timeline HR Estimation: Kalman vs Median vs Raw FFT (Stage I-2)', color='white', fontsize=12, y=0.98, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        path = os.path.join(self.output_dir, f'plot_kalman_history_{ts}.png')
        canvas.print_figure(path, dpi=110)
        print(f"  Kalman hist -> {path}")