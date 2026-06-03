import os
import matplotlib.pyplot as plt
from plot_utils import ieee_fig, STYLES, load_pipeline_state, get_figures_dir

def main():
    state = load_pipeline_state()
    cs_data = state['cost_sensitivity']
    lams = cs_data['multipliers']
    csens = cs_data['costs']
    
    fig, ax = ieee_fig(h=2.2)
    
    for nm in ['RF', 'XGBoost', 'AADNN', 'Hybrid']:
        st = STYLES[nm]
        cv = [csens[l][nm] / 1000 for l in lams]
        ax.plot(lams, cv, linestyle=st['ls'], color=st['color'], marker=st['marker'],
                lw=st['lw'], ms=st['lw'] * 3.5, label=nm)
        
    ax.set(xlabel='$\\lambda_{FN}$ (FN cost multiplier)', ylabel='Total Cost (\\$K)')
    ax.legend(fontsize=6)
    
    fig_dir = get_figures_dir()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig12_cost.png'))
    plt.close()
    print("fig12_cost.png generated.")

if __name__ == '__main__':
    main()
