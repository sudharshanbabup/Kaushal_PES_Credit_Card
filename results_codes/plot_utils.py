import os
import pickle
import matplotlib.pyplot as plt

# Constant for IEEE single-column width in inches
COL_W = 3.487

# B&W friendly color palette and markers
STYLES = {
    'RF':       {'color': '#2c7bb6', 'marker': 'o', 'ls': '--', 'lw': 0.9},
    'XGBoost':  {'color': '#d7191c', 'marker': 's', 'ls': '-.', 'lw': 0.9},
    'AADNN':    {'color': '#fdae61', 'marker': '^', 'ls': ':',  'lw': 0.9},
    'AE':       {'color': '#abd9e9', 'marker': 'D', 'ls': '--', 'lw': 0.9},
    'Hybrid':   {'color': '#000000', 'marker': '*', 'ls': '-',  'lw': 1.4},
}

def set_ieee_style():
    """
    Apply standard IEEE publication styling parameters to Matplotlib.
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5,
        'legend.fontsize': 7,
        'figure.dpi': 600,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'axes.grid': False,
        'axes.linewidth': 0.6,
        'lines.linewidth': 1.0,
        'lines.markersize': 4,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'legend.framealpha': 1.0,
        'legend.edgecolor': '0.8',
        'legend.frameon': True,
        'legend.handlelength': 1.8,
    })

def ieee_fig(w=COL_W, h=None, ratio=0.75):
    """
    Create a matplotlib figure sized for IEEE publications.
    """
    set_ieee_style()
    if h is None:
        h = w * ratio
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax

def load_pipeline_state():
    """
    Load the cached pipeline state. If the cache does not exist,
    run the pipeline automatically to generate the cache.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(base_dir, 'cache', 'pipeline_state.pkl')
    
    if not os.path.exists(cache_path):
        print("Pipeline state cache not found. Running the pipeline first to build cache...")
        from pipeline import run_pipeline
        # Run pipeline to generate cache
        return run_pipeline(force_rerun=False)
        
    with open(cache_path, 'rb') as f:
        return pickle.load(f)

def get_figures_dir():
    """
    Get the path to the figures directory.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(base_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir
