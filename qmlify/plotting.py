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

def plot_calibration(solvent_dict=None,
                     complex_dict=None,
                     timestep_in_fs = 2.0,
                     fig_width=8.5,
                     fig_height=7.5,
                     suptitle = None,
                     plot_name = "calibration.pdf"):
    """
    make a calibration plot for the work standard deviation w.r.t. annealing time; write plot to disk
    NOTE : each dict is of the form returned by `qmlify.analyis. extract_work_calibrations`

    arguments
        solvent_dict : dict, default None
            dict of [annealing_steps: work_array]
        complex_dict : dict, default None
            dict of [annealing_steps : work_array]
        timestep_in_fs : float, default 2.0
            timestep size in fs
        fig_width : float, default 8.5
            figure width in inches
        fig_height : float, default 7.5
            figure height in inches
        suptitle : str, default None
            the sup title of the plot
        plot_name : str
            name to write to (end in .png or .pdf)
    """
    from qmlify.analysis import bootstrap, compute_CIs
    fig = plt.figure(figsize=(fig_width, fig_height))
    plot_solvent=True if solvent_dict is not None else False
    plot_complex=True if complex_dict is not None else False

    if plot_solvent and plot_complex:
        complex_ax = fig.add_subplot(2,2,1)
        solvent_ax = fig.add_subplot(2,2,3)
        plotter_ax = fig.add_subplot(1,2,2)
    elif plot_solvent and not plot_complex:
        solvent_ax = fig.add_subplot(1,2,1)
        plotter_ax = fig.add_subplot(1,2,2)
    elif plot_complex and not plot_solvent:
        complex_ax = fig.add_subplot(1,2,1)
        plotter_ax = fig.add_subplot(1,2,2)

    #define an offset...
    if plot_solvent:
        solvent_keys = list(solvent_dict.keys())
        offset = np.mean(solvent_dict[solvent_keys[0]])

    #set xlims
    mins, maxs = [], []
    if plot_solvent:
        mins += [np.min(val) for val in solvent_dict.values()]
        maxs += [np.max(val) for val in solvent_dict.values()]
    if plot_complex:
        mins += [np.min(val) for val in complex_dict.values()]
        maxs += [np.max(val) for val in complex_dict.values()]

    xlims = (min(mins), max(maxs))



    #set labels..
    if plot_solvent:
        solvent_ax.set_xlabel(f"work [kT]")
        solvent_ax.get_yaxis().set_ticks([])


    if plot_solvent:
        color_counter = 0
        for annealing_step in solvent_dict.keys():
            sns.distplot(solvent_dict[annealing_step] - offset, color = cycle[color_counter], label = f"{annealing_step*timestep_in_fs/1000} ps", ax=solvent_ax)
            color_counter +=1
        solvent_ax.legend()
        solvent_ax.set_xlim(xlims[0] - offset - 5, xlims[1] - offset + 5)
        solvent_ax.set_title(f"solvent")
        solvent_ax.set_ylabel("$P(work)$")
        if plot_complex: complex_ax.xaxis.set_ticklabels([])

    if plot_complex:
        complex_ax.get_yaxis().set_ticks([])
        color_counter = 0
        for annealing_step in complex_dict.keys():
            sns.distplot(complex_dict[annealing_step] - offset, color = cycle[color_counter], label = f"{annealing_step*timestep_in_fs/1000} ps", ax=complex_ax)
            color_counter +=1
        complex_ax.legend()
        complex_ax.set_xlim(xlims[0] - offset - 5, xlims[1] - offset + 5)
        complex_ax.set_title(f"complex")
        complex_ax.set_ylabel("$P(work)$")

    #plotter ax
    if plot_solvent:
        work_stddev = [np.std(vals) for vals in solvent_dict.values()]
        bounds = [compute_CIs(bootstrap(val, np.std, num_resamples=10000), alpha=0.95) for val in solvent_dict.values()]

        for idx in range(len(solvent_dict)):
            label = 'solvent' if idx==0 else None
            y = work_stddev[idx]
            fix_bounds = np.array(bounds[idx])
            fix_bounds[0] = y - fix_bounds[0]
            fix_bounds[1] = fix_bounds[1] - y
            plotter_ax.errorbar(list(solvent_dict.keys())[idx]*timestep_in_fs/1000,
                                y,
                                ls='None',
                                marker = 'o',
                                color = cycle[idx],
                                yerr = fix_bounds.reshape(2,1),
                                alpha=0.5,
                                markersize=10,
                                elinewidth=3,
                                label=label)
            if idx==0: plotter_ax.legend()
        plotter_ax.set_xscale('log')
        plotter_ax.set_xlabel(f"annealing time [ps]")
        plotter_ax.set_ylabel(f"work standard deviation [kT]")

    if plot_complex:
        work_stddev = [np.std(vals) for vals in complex_dict.values()]
        bounds = [compute_CIs(bootstrap(val, np.std, num_resamples=10000), alpha=0.95) for val in complex_dict.values()]

        for idx in range(len(complex_dict)):
            label = 'complex' if idx==0 else None
            y = work_stddev[idx]
            fix_bounds = np.array(bounds[idx])
            fix_bounds[0] = y - fix_bounds[0]
            fix_bounds[1] = fix_bounds[1] - y
            plotter_ax.errorbar(list(complex_dict.keys())[idx]*timestep_in_fs/1000,
                                y,
                                ls='None',
                                marker = '^',
                                color = cycle[idx],
                                yerr = fix_bounds.reshape(2,1),
                                alpha=0.5,
                                markersize=10,
                                elinewidth=3,
                                label = label)
            if idx==0: plotter_ax.legend()
        plotter_ax.set_xscale('log')
        plotter_ax.set_xlabel(f"annealing time [ps]")
        plotter_ax.set_ylabel(f"work standard deviation [kT]")

    plt.tight_layout()

    if suptitle is not None:
        fig.suptitle(suptitle)

    fig.savefig(plot_name)
