import os
import matplotlib.pyplot as plt
from plot_utils import ieee_fig, load_pipeline_state, get_figures_dir

def main():
    state = load_pipeline_state()
    nfr = state['novel_fraud']
    
    fig, ax = ieee_fig(h=2.2)
    
    nfn = list(nfr.keys())
    nfv = [nfr[n] * 100 for n in nfn]
    colors_nf = ['#2c7bb6', '#d7191c', '#fdae61', '#abd9e9', '#000000']
    
    bars = ax.bar(nfn, nfv, color=colors_nf, edgecolor='none', width=0.55)
    for b, v in zip(bars, nfv):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f'{v:.1f}%', ha='center', fontsize=7, fontweight='bold')
        
    ax.set(ylabel='Detection Rate (%)')
    ax.set_ylim(0, max(nfv) * 1.2)
    
    fig_dir = get_figures_dir()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig10_novel.png'))
    plt.close()
    print("fig10_novel.png generated.")

if __name__ == '__main__':
    main()
