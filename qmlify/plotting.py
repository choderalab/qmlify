"""
plotting utilities; best to import * to use utilities in this module for full functionality
"""
#####Imports#####
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import numpy as np

#####Defaults#####
mpl.rcParams['axes.formatter.useoffset'] = False
VERY_SMALL_SIZE = 8
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 24
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes\n",
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title\n",
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels\n",
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels\n",
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels\n",
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize\n",
plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title"

cycle = list(plt.rcParams['axes.prop_cycle'].by_key().values())[0]

def generate_work_distribution_plots(work_dict,
                                     BAR_dict,
                                     out_prefix,
                                     ligands_per_plot = 4,
                                     width = 8.5,
                                     height_unit = 5,
                                     legend=False):
    """
    make a series of plots of the forward/backward work distributions and BAR values (with errors) for each phase

    arguments
        work_dict : dict
            dict of output type qmlify.analysis.fully_aggregate_work_dict
        BAR_dict : dict
            dict of output type qmlify.analysis.compute_BAR
        out_prefix : str
            string that is the prefix to lig{i}to{j}.pdf
        ligands_per_plot : int, default 4
            number of columns (each representing a different ligand, complex and solvent) in each image
        width : float, default 8.5
            width of each image in inches
        height_unit : float, default 5
            height of each image in inches
        legend : bool, default False
            whether to show legend of each plot

    """
    unique_ligands = sorted(list(work_dict.keys()))
    full_divisions = len(unique_ligands)//ligands_per_plot
    remainder = len(unique_ligands) % ligands_per_plot
    num_plots = full_divisions + 1 if remainder !=0 else full_divisions
    print(f"generating {num_plots} plots...")
    ligand_counter = 0

    for plot_idx in range(num_plots): #iterate over all plots
        fig = plt.figure(figsize=(width, height_unit))
        start = ligand_counter
        end = start + ligands_per_plot if plot_idx != num_plots else start + remainder
        print(f"plotting ligands: {start} through {end}")
        for ligand_idx, ligand in enumerate(unique_ligands[start:end]):
            forward_solvent_work_hists = work_dict[ligand]['solvent']['forward']
            backward_solvent_work_hists = work_dict[ligand]['solvent']['backward']
            forward_complex_work_hists = work_dict[ligand]['complex']['forward']
            backward_complex_work_hists = work_dict[ligand]['complex']['backward']

            #detemine xlims
            abs_min = min([np.min(entry) for entry in [forward_solvent_work_hists, -backward_solvent_work_hists, forward_complex_work_hists, -backward_complex_work_hists]])
            abs_max =  max([np.max(entry) for entry in [forward_solvent_work_hists, -backward_solvent_work_hists, forward_complex_work_hists, -backward_complex_work_hists]])


            #BAR results:
            complex_BARs = BAR_dict[ligand]['complex']
            solvent_BARs = BAR_dict[ligand]['solvent']

            #complex
            ax_complex = fig.add_subplot(2, ligands_per_plot, ligand_idx+1)
            ax_complex.set_title(f"ligand {ligand}: complex")
            ax_complex.get_yaxis().set_ticks([])
            ax_complex.get_xaxis().set_ticklabels([])

            offset = complex_BARs[0][0]

            rel_min = abs_min - offset - 5
            rel_max = abs_max - offset + 5
            ax_complex.set_xlim(rel_min, rel_max)

            for entry in complex_BARs:
                dg, ddg = entry
                ax_complex.axvline(dg - offset, color='k')
                ax_complex.axvline(dg - offset + ddg, color = 'gray', ls = '--')
                ax_complex.axvline(dg - offset - ddg, color = 'gray', ls = '--')


            counter=0
            for entry in forward_complex_work_hists:
                if counter==0 and ligand_idx==0:
                    label = 'complex: forward'
                else:
                    label = None
                sns.distplot(entry - offset, color = cycle[0], ax=ax_complex, label=label)
                counter+=1

            counter=0
            for entry in backward_complex_work_hists:
                if counter==0 and ligand_idx==0:
                    label = 'complex: backward'
                else:
                    label = None
                sns.distplot(-entry - offset, color = cycle[1], ax=ax_complex, label=label)
                counter+=1

            if legend: plt.legend()

            #solvent
            ax_solvent = fig.add_subplot(2, ligands_per_plot, ligand_idx+ligands_per_plot+1)
            ax_solvent.set_title(f"ligand {ligand}: solvent")
            ax_solvent.get_yaxis().set_ticks([])
            ax_solvent.set_xlabel(f"work [kT]")

            ax_solvent.set_xlim(rel_min, rel_max)

            for entry in solvent_BARs:
                dg, ddg = entry
                ax_solvent.axvline(dg - offset, color='k')
                ax_solvent.axvline(dg - offset + ddg, color = 'gray', ls = '--')
                ax_solvent.axvline(dg - offset - ddg, color = 'gray', ls = '--')

            counter=0
            for entry in forward_solvent_work_hists:
                if counter==0 and ligand_idx==0:
                    label = 'solvent: forward'
                else:
                    label = None
                sns.distplot(entry - offset, color = cycle[0], ax=ax_solvent, label=label)
                counter+=1

            counter=0
            for entry in backward_solvent_work_hists:
                if counter==0 and ligand_idx==0:
                    label = 'solvent: backward'
                else:
                    label = None
                sns.distplot(-entry - offset, color = cycle[1], ax=ax_solvent, label=label)
                counter+=1

            if legend: plt.legend()
        plt.tight_layout()
        fig.savefig(f"{out_prefix}.lig{start}to{end}.pdf")
        ligand_counter += ligands_per_plot
