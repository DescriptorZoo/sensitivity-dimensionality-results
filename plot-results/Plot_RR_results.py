import os
import sys
import math
import numpy as np
import copy
import types
import time
import json
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_ticker
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from cycler import cycler

PI = 3.14159265359
unit_BOHR = 0.52917721067

def fmt_table(table):
    return '\n'.join([' '.join(['{: ^6}'.format(x) for x in row]) for row in table])

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class results_file(object):
    def __init__(self, filename=None):
        self.updated = False
        self.dct = {}
        if filename is not None:
            self.filename = filename
        else:
            self.filename = 'results.json'
        self.file_exists = os.path.isfile(self.filename)
        if self.file_exists:
            self.load(filename)

    def add(self, key, value): 
        if key not in self.dct:
            self.dct[key] = value
            self.updated = True

    def write(self, filename=None):
        if self.updated:
            if filename is None:
                fname = self.filename
            else:
                fname = filename
            print('Updating '+fname+' file...')
            with open(fname, 'w') as fp: 
                json.dump(self.dct, fp, indent=4, cls=NumpyEncoder)

    def load(self, filename=None):
        if filename is None:
            fname = self.filename
        else:
            fname = filename
        print('Loading '+fname+' file...')
        #try:
        with open(fname, 'r') as fp: 
            self.dct = json.load(fp)
        #except:
        #    if self.dct:
        #        pass
        #    else:
        #        self.dct = {}

def load_json(filename):
    dct = {}
    with open(filename, 'r') as fp: 
        dct = json.load(fp)
    return dct 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyses on Sensitivity and Rotational Invariance of Descriptors.")
    group = parser.add_mutually_exclusive_group()
    parser.add_argument("-i","--filename", type=str, required=False, default='../results/RR_results.json', 
            help="File path and name for the structures.")
    parser.add_argument("-o","--outfile", type=str, required=False, default=None,
            help="File path and name for the outputs.")
    parser.add_argument("-c", "--color", action="store_true",
            help="Plot atom sensitivities.")
    parser.add_argument("-g", "--grid", action="store_true",
            help="Plot grids.")
    parser.add_argument("-s", "--seaborn", action="store_true",
            help="Plot atom sensitivities.")
    parser.add_argument("-LR", "--LR", action="store_true",
            help="Plot LR results.")
    parser.add_argument("-n","--nfeatures", type=int, required=False, default=65,
            help="Plot upto the given number of features.")
    parser.add_argument("-y","--y", type=int, required=False, default=2,
            help="Plot upto the given number of features.")
    parser.add_argument("-plotlabels","--plotlabels", type=str, required=False, default="RR,KRR",
            help="File path and name for the outputs.")
    parser.add_argument("-plotname","--plotname", type=str, required=False, default="Si",
            help="File path and name for the outputs.")
    args = parser.parse_args()

    print('  Reading JSON results file:',args.filename)
    resfile = results_file(args.filename)
    sys.stdout.flush()
    numplots = len(resfile.dct.keys())
    lr_labels = [str(s) for s in args.plotlabels.split(",")]
   
    font = {'family': 'monospace',
            'color':  'black',
            'weight': 'normal',
            'size': 16,
            }
    fontx = {
            'family': 'sans-serif',
            'size': 12,
            }
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['mathtext.rm'] = 'Times New Roman'
    mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
    mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'

    fontp = FontProperties()
    fontp.set_family('sans')
    fontp.set_name('Times New Roman')
    fontp.set_size(16)
    fontp.set_weight('light')
    fontp.set_style('italic')
    if args.color:
        color_codes = ['darkgreen',  # x
                       'darkred',    # y
                       'mediumblue', # z
                       'dodgerblue', # -z
                       'red',        # -y
                       'limegreen',  # -x
                       'darkorange', 'purple',
                       'pink','lightgreen','orangered','lightblue','yellow',
                       'red','blue','darkorange', 'purple',
                       'pink','lightgreen','darkgrey','lightblue','yellow']
        color_maps = [range(numplots)]
    else:
        color_codes = plt.cm.Set1(np.arange(0,numplots*2)) 
        color_maps = [3,7,0,1,4,2,6,8,0,3]
    if args.seaborn:
        mpl.style.use('seaborn')
    line_types = ['-','-','-','-','-','-','-','-','--','-','--','-','--','-','--','-','--','-','--','-','--','-','--']
    line_sizes = [[3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.5,3.5,3.5,3.5,3.5],
                  [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,3.5,3.5,3.5,3.5]]
    select_plots = [[],[],[],[]]
    for p in resfile.dct.keys():
        if args.plotname in resfile.dct[p]:
            if 'training_LR+L2_RMSE_eV_atom' in resfile.dct[p][args.plotname].keys():
                select_plots[0].append(p)
                print(p+" added to LSQ+L2 training plots.")
            if 'test_LR+L2_RMSE_eV_atom' in resfile.dct[p][args.plotname].keys():
                select_plots[0].append(p)
                print(p+" added to LSQ+L2 test plots.")
            if 'training_KRR_RMSE_eV_atom' in resfile.dct[p][args.plotname].keys():
                select_plots[1].append(p)
                print(p+" added to KRR training plots.")
            if 'test_KRR_RMSE_eV_atom' in resfile.dct[p][args.plotname].keys():
                select_plots[1].append(p)
                print(p+" added to KRR test plots.")
    numplots = len(lr_labels)
    plotrows = int(numplots/args.y)
    if args.y > 1:
        plotextras = numplots%args.y
    else:
        plotextras = 0
    extraflag=False
    if plotextras > 0:
        plotrows += 1
        extraflag=True
    print('numplots:',numplots)
    print('plotrows:',plotrows)
    print('plotextras:',plotextras)
    fig = plt.figure(figsize=(args.y*10*0.7, 2*4*plotrows*0.7), dpi= 80,)
    gs = []
    pbottom = [0.55, 0.09]
    ptop = [0.98, 0.52]
    pleft = [0.06, 0.06]
    pright = [0.99, 0.99]
    plotid = 0
    for gridno in range(2):
        plotno=0
        plotrow=0
        gs.append(fig.add_gridspec(nrows=plotrows, ncols=args.y, 
                                   top=ptop[gridno], bottom=pbottom[gridno],
                                   left=pleft[gridno], right=pright[gridno],
                                   wspace=0.02, hspace=0.08))
        for pltno in range(numplots):
            plotno_id = plotno%args.y
            if (plotno+1)%args.y==1:
                plotrow += 1
            print("Adding subplot: ",plotrow-1,plotno_id)
            ax = fig.add_subplot(gs[gridno][plotrow-1,plotno_id])
            ax.plot((1.0E-14,1.0), (1.0E-16, 1.0E-16), 'k-.', lw=2)
            figleg_handles = []
            figleg_labels = []
            ci = 0
            for k, v in resfile.dct.items():
                if k not in select_plots[plotno]:
                    continue
                desc_label = k
                if args.plotname not in v:
                    continue
                else:
                    if('training_LR+L2_RMSE_eV_atom' not in v[args.plotname]):
                        continue
                    else:
                        print(k)
                res = v
                for ri, r in enumerate(res.keys()):
                    addthis=True
                    if r not in args.plotname:
                        continue
                    ydata = []
                    if gridno < 1:
                        ydata.append(np.array(res[r]['training_LR+L2_RMSE_eV_atom']))
                        ydata.append(np.array(res[r]['test_LR+L2_RMSE_eV_atom']))
                    else:
                        ydata.append(np.array(res[r]['training_KRR_RMSE_eV_atom']))
                        ydata.append(np.array(res[r]['test_KRR_RMSE_eV_atom']))
                    if plotno < 1:
                        if args.nfeatures <= len(ydata[0]):
                            x_percents = np.arange(1,args.nfeatures+1)
                        else:
                            x_percents = np.arange(1,len(ydata[0])+1)
                    else:
                        feature_last = 100.0
                        if 'ACSF-X' in desc_label:
                            feature_last = 100.0*(res[r]['feature_num_reduced']/res[r]['feature_num'])
                        feat_num = res[r]['feature_num_reduced']
                        x_percents = np.arange(1,feat_num+1)*feature_last/feat_num
                    xdata=[x_percents for yi in range(len(ydata))]
                    ldata=desc_label
                    if addthis:
                        colorid = color_maps[int(ci)]
                        rp = [ydata[yy][:len(xdata[yy])] for yy in range(len(ydata))]
                        for yi, yy in enumerate(rp):
                            thisplot, = ax.plot(xdata[yi], rp[yi], color=color_codes[colorid], lw=line_sizes[yi][int(ci)], linestyle=line_types[int(ci)], label=ldata)
                            if yi == 0:
                                figleg_handles.append(thisplot)
                                figleg_labels.append(ldata)
                        ci += 1
                ax.set_yscale('log')
                if args.grid:
                    plt.grid(True)
                plt.tick_params(
                    bottom=True,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    left=True,      # ticks along the bottom edge are off
                    right=False,         # ticks along the top edge are off
                    labelleft=False,
                    labelbottom=False) # labels along the bottom edge are off
                # Hide the right and top spines
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                if (plotno+1)%args.y==1 or args.y==1:
                    ax.set_ylabel('RMSE (eV/atom)', fontsize=14)
                    plt.tick_params(
                        axis='both',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        left=True,      # ticks along the bottom edge are off
                        right=False,         # ticks along the top edge are off
                        labelleft=True,
                        bottom=True,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelbottom=False) # labels along the bottom edge are off
                if extraflag is False and (plotrow==plotrows or plotrow==2) and gridno >0:
                    if plotno < 1: 
                        ax.set_xlabel(r'$\mathrm{Number\,\,of\,\,Features\,}$', fontproperties=fontp)
                    else:
                        ax.set_xlabel(r'$\mathrm{Percentage\,\,of\,\,Features\,}(\%)$', fontproperties=fontp)
                    plt.tick_params(
                        axis='both',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom=True,      # ticks along the bottom edge are off
                        left=True,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelbottom=True) # labels along the bottom edge are off
                if extraflag and plotextras!=0 and plotrow==plotrows-1 and (plotno+1)%args.y not in [i+1 for i in range(plotextras)] and gridno >0:
                    if plotno < 1: 
                        ax.set_xlabel(r'$\mathrm{Number\,\,of\,\,Features\,}$', fontproperties=fontp)
                    else:
                        ax.set_xlabel(r'$\mathrm{Percentage\,\,of\,\,Features\,}(\%)$', fontproperties=fontp)
                    plt.tick_params(
                        axis='both',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom=True,      # ticks along the bottom edge are off
                        left=True,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelbottom=True) # labels along the bottom edge are off
                if extraflag and plotextras!=0 and plotrow==plotrows and (plotno+1)%args.y in [i+1 for i in range(plotextras)] and gridno >0:
                    if plotno < 1: 
                        ax.set_xlabel(r'$\mathrm{Number\,\,of\,\,Features\,}$', fontproperties=fontp)
                    else:
                        ax.set_xlabel(r'$\mathrm{Percentage\,\,of\,\,Features\,}(\%)$', fontproperties=fontp)
                    plt.tick_params(
                        axis='both',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom=True,      # ticks along the bottom edge are off
                        left=True,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelbottom=True) # labels along the bottom edge are off
                ax.xaxis.set_major_locator(plt.MultipleLocator(5))
                ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
                if plotno < 1: 
                    plt.xlim((-1,args.nfeatures+1))
                else:
                    plt.xlim((-1,101))
            if gridno < 1: 
                plt.ylim((1.0E-2,1.0E+1))
            else:
                plt.ylim((2.0E-3,1.0E+1))
            addtext=True
            if (gridno==0 and plotno==0):
                first_legend = ax.legend(handles=[ l for l in figleg_handles[:3]],
                                         fontsize=14, loc=(0.01,-0.005), framealpha=0,
                                         handletextpad=0.3,labelspacing=0.25)
                ax1 = plt.gca().add_artist(first_legend)
                second_legend = plt.legend(handles=[l for l in figleg_handles[3:5]],
                                           fontsize=14, loc=(0.3,-0.005), framealpha=0,
                                           handletextpad=0.3,labelspacing=0.25)
                ax2 = plt.gca().add_artist(second_legend)
                plt.legend(handles=[l for l in figleg_handles[5:]],
                           fontsize=14, loc=(0.6,-0.005), framealpha=0,
                           handletextpad=0.3,labelspacing=0.25)
            if addtext:
                print("Labeling plot ",plotno+1," with ",lr_labels[gridno])
                plt.text(0.5-0.021*len(lr_labels[gridno])/2, 0.93, lr_labels[gridno], fontdict=font, transform=ax.transAxes)
                if (plotno+1)%args.y!=1 and args.y>1:
                    plt.text(-0.04, 0.93, chr(plotid+97)+')', fontdict=font, transform=ax.transAxes)
                if (plotno+1)%args.y==1 or args.y==1:
                    plt.text(-0.12, 0.93, chr(plotid+97)+')', fontdict=font, transform=ax.transAxes)
            plotno+=1
            plotid+=1
    #if args.outfile is not None:
    #    base = os.path.basename(args.outfile)
    #else:
    #    base = os.path.basename(args.filename)
    #filebase = os.path.splitext(base)[0]
    plt.tight_layout()
    fig.savefig('figure9.pdf',dpi= 100)
    print('Figure9 is generated')
    
