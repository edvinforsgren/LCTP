from analysis import utils as ut
import os
import yaml
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns
import argparse
import numpy as np
import time

def plot_predictions_by_compound_without_axis(full_df, moas, pred_labels, plot_how=None, save_dir=None, figsize=(12, 12), cmpd_col = 'Metadata_cmpd_cmpdname', format='png', nrows=3, ncols=4):
    """Plot prediction curves for each compound
    
    Args:
        full_df (pd.DataFrame): DataFrame containing predictions
        moas (list): List of MOA classes
        pred_labels (list): List of prediction column names
        plot_platewise (bool): Whether to plot different plates with different line styles
        save_dir (str): Directory to save plots (optional)
    """

    # Define the colors
    colors = [
        "#76c76b", "#2e8b57",  # Adjusted light and dark green
        "#87ceeb", "#1e3a8a",  # Adjusted light and slightly brighter dark blue
        "#ff9999", "#b22222",  # Adjusted light and slightly brighter dark red
        "#ffcc33", "#ff8c00",  # Adjusted light and dark yellow/orange
        "#bfbfbf", "#696969",  # Adjusted light and dark gray
        "#c9a0dc", "#6a0dad"   # Adjusted light and slightly brighter dark purple
    ]
    # Create the categorical colormap
    cmap = ListedColormap(colors)
    color_map = {label: cmap(i) for i, label in enumerate(pred_labels)}
    

    for cl in moas:
        sort_df = full_df[full_df['Metadata_cmpd_moa_group'] == cl].copy()
        sort_df['Metadata_hours'] = pd.to_numeric(sort_df['Metadata_hours'], errors='coerce')
        sort_df = sort_df.dropna(subset=['Metadata_hours']).sort_values('Metadata_hours')

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 3))
        axes = axes.flatten()
        ## Cleaning up the compound names
        all_compounds = sorted(sort_df[cmpd_col].unique().tolist(), key=str.casefold)
        fixed_all_compounds = [None] * len(all_compounds)
        for i, compound in enumerate(all_compounds):
            compound = compound[0].upper() + compound[1:]
            if not compound.startswith('('):
                compound = compound.split(' (')[0].strip()
            if compound == 'ALBENDAZOLE':
                compound = 'Albendazole'
            if compound == 'MEBENDAZOLE':
                compound = 'Mebendazole'
            fixed_all_compounds[i] = compound
        fixed_all_compounds = sorted(fixed_all_compounds, key=str.casefold)
        
        
        for i, compound in enumerate(all_compounds):
            ax = axes[i]
            class_df = sort_df[sort_df[cmpd_col] == compound]
            copy_df = class_df.copy()

            _plot_predictions(copy_df, pred_labels, color_map, ax, alpha=0.8)
                
            
            compound = fixed_all_compounds[i]
            print(compound)
            if len(compound.split(' ')) > 2:
                compound = ' '.join(compound.split(' ')[:2])
            ax.set_title(f"{compound}", fontsize=24)
            ax.set_xticks([])
            ax.set_yticks([])
            
            ax.set_xlim((0, 72))
            ax.set_ylim((-0.5, 2))
            
            ax.set_ylabel('')
            
            ax.set_xlabel('')
            _, labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()
        if i < 11:
            i += 1
            ax = axes[i]
            ax.axis('off')
        # Adjust layout to create space for the legend
        plt.tight_layout()
                       
        # Create custom legend handles with desired linewidth
        custom_handles = [
            matplotlib.lines.Line2D([0], [0], color=color_map[label], lw=5) for label in labels
        ]
        yield fig, ax
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f'{save_dir}/{cl}_nn_eq_scores.{format}', bbox_inches='tight', dpi=400)
            
    # Add legend below the figure
    fig, ax = plt.subplots(figsize=(12, 1))
    labels = [label.split('_')[0] + '_' + label.split('_')[-1]  for label in pred_labels]
    fig.legend(custom_handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), 
    ncol=len(pred_labels) // 2, prop={'size': 14})
    ax.axis('off')
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}/legend.{format}', bbox_inches='tight', dpi=300)    


def _plot_predictions(df, pred_labels, color_map, ax, alpha=0.8):
    """Helper function for plotting predictions without plate-wise styling"""
    for pred_label in pred_labels:
        for _, group in df.groupby('Metadata_Well'):
            smoothed_values = ut.noise_reduction(group[pred_label].values)
            df.loc[group.index, f'smoothed_{pred_label}'] = smoothed_values
        
        sns.lineplot(data=df,
                    x='Metadata_hours',
                    y=f'smoothed_{pred_label}',
                    label=pred_label,
                    errorbar='sd',
                    ax=ax,
                    color=color_map[pred_label],
                    alpha=alpha)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

        

def plot_all_compounds_without_axis(all_figs, moas, save_dir=None, figsize=(8, 7.5), format='pdf'):
    fig, axes = plt.subplots(3, 2, figsize=figsize)  # Create subplots
    axes = axes.flatten()

    for i, cl in enumerate(moas):
        print(cl)
        ax = axes[i]

        # Render figure to a canvas and extract as an image
        fig_canvas = all_figs[i]
        fig_canvas.set_dpi=300
        canvas = FigureCanvas(fig_canvas)
        canvas.draw()
        img = np.array(canvas.renderer.buffer_rgba())

        # Display the image in the corresponding subplot
        ax.imshow(img)
        ax.set_title(cl, fontsize=18)
        ax.axis('off')  # Hide axis for clean display

    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}/all_nn_eq_scores.{format}', bbox_inches='tight', dpi=300, format=format)

    plt.show()



if __name__  == "__main__":
    # Start times to measure runtime
    start = time.time()
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to the YAML config file.")
    args = parser.parse_args()
    config_file_path = args.config_file

    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)

    # Get parameters from config    
    cell = config['data']['cell']
    save_path = config['data']['save_path']
    batch_size = config['model']['batch_size']
    normalize_str = config['model']['normalize']
    learning_rate = config['model']['learning_rate']
    loss_function_str = config['model']['loss_function']


        
    save_dir = f'{save_path}bs{batch_size}_lr{learning_rate}_loss{loss_function_str}_norm{normalize_str}'
    file_name = f'{cell}_dino_ts'
    os.makedirs(save_dir, exist_ok=True)
    full_df = pd.read_parquet(f'{save_dir}/{file_name}_cmpd_nn_eq_pred.parquet.gzip')
    full_dmsos = pd.read_parquet(f'{save_dir}/{file_name}_dmso_nn_eq_pred.parquet.gzip')
    
    moas = [moa for moa in full_df.Metadata_cmpd_moa_group.unique().tolist() 
            if (moa and 'dmso' not in str(moa))]
    moas.sort()
    
    
    # Concatenate all prediction dataframes
    full_df.sort_values('Metadata_cmpd_cmpdname', inplace=True)
    # Flatten prediction labels from nested list
    flatten_pred_labels = [f'{c}_pred_{t}'  for c in moas for t in ['early', 'late']]
    all_figs, all_axes = [], []
    # Plot predictions
    figs_ax = []
    save_dir = f'{save_dir}/nn_eq_plots/{cell}-no_axis'
    os.makedirs(save_dir, exist_ok=True)
    for fig_ret, ax_ret in plot_predictions_by_compound_without_axis(
        full_df=full_df,  
        moas=moas,
        pred_labels=flatten_pred_labels,
        plot_how=None,
        figsize=(16, 9),
        cmpd_col='Metadata_cmpd_oldcmpdname',
        save_dir=save_dir,
        format='png',
        nrows=3,
        ncols=4
    ):
        all_figs.append(fig_ret)
        all_axes.append(ax_ret)
    
    plot_all_compounds_without_axis(all_figs=all_figs, moas=moas, save_dir=f'{save_dir}/', format='svg')

    stop_time = time.time() - start

    print(f"Finished {cell}")
    # Print runtime
    print(f"Runtime: {time.time() - start}")