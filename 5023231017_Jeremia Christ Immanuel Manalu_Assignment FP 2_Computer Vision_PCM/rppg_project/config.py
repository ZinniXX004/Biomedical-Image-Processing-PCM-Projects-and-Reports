# config.py

"""
Default configuration parameters for the rPPG application.
This parameter will be loaded when the first time the application is run, and can be modified by the user through the GUI.
"""

DEFAULT_CONFIG = {
    "camera_index": 0,
    "fps": 30.0,
    "buffer_seconds": 10.0,
    "min_ready_sec": 5.0,
    "update_interval": 15,
    "plot_interval": 15,
    "bpf_low_hz": 0.75,
    "bpf_high_hz": 2.5,
    "sg_window": 9,
    "sg_poly": 3,
    "rppg_method": "chrom",
    "bpm_min": 45.0,
    "bpm_max": 150.0,
    "kalman_q_bpm": 0.5,
    "kalman_q_trend": 0.1,
    "kalman_r": 8.0,
    "output_dir": "rppg_output",
    "auto_save": True,
}