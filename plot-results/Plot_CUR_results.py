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
from matplotlib.font_manager import FontProperties
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
    parser.add_argument("-i","--filename", type=str, required=False, default='../results/CUR_results.json', 
            help="File path and name for the structures.")
    parser.add_argument("-o","--outfile", type=str, required=False, default=None,
            help="File path and name for the outputs.")
    parser.add_argument("-c", "--color", action="store_true",
            help="Plot atom sensitivities.")
    parser.add_argument("-g", "--grid", action="store_true",
            help="Plot grids.")
    parser.add_argument("-s", "--seaborn", action="store_true",
            help="Plot atom sensitivities.")
    parser.add_argument("-y","--y", type=int, required=False, default=2,
            help="File path and name for the outputs.")
    args = parser.parse_args()

    print('  Reading JSON results file:',args.filename)
    resfile = results_file(args.filename)
    sys.stdout.flush()
    numplots = len(resfile.dct.keys())
   
    font = {
            'family': 'monospace',
            #'name': 'Times New Roman',
            'color':  'black',
            #'weight': 'bold',
            'weight': 'normal',
            'size': 16,
            }
    fontx = {
            'family': 'sans-serif',
            #'name': 'Times New Roman',
            #'color':  'black',
            #'weight': 'bold',
            'size': 12,
            }
    #from matplotlib import rc
    #rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    ## for Palatino and other serif fonts use:
    #rc('font',**{'family':'serif','serif':['Palatino']})
    #rc('text', usetex=True)
    #mpl.rcParams['text.usetex'] = True
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['mathtext.rm'] = 'Times New Roman'
    mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
    mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'

    fontp = FontProperties()
    fontp.set_family('sans')
    fontp.set_name('Times New Roman')
    #fontp.set_family('sans-serif')
    #fontp.set_name('Times')
    fontp.set_size(16)
    #fontp.set_weight('normal')
    fontp.set_weight('light')
    fontp.set_style('italic')
    #fontp.set_style('oblique')
    #font = {'family': 'monospace',
    #        'color':  'black',
    #        'weight': 'bold',
    #        'size': 16,
    #        }
    #color_codes = ['red','blue','darkorange', 'purple',
    #               'pink','lightgreen','darkgrey','lightblue','yellow',
    #               'red','blue','darkorange', 'purple',
    #               'pink','lightgreen','darkgrey','lightblue','yellow']
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
        color_codes = plt.cm.Paired(np.arange(0,numplots*2)) 
        color_maps = [7,3,1,5,9,4,6,8,0,3]
    if args.seaborn:
        mpl.style.use('seaborn')
    line_types = ['-','-','-','-','--','--','--','-','--','-','--','-','--','-','--','-','--','-','--','-','--','-','--']
    line_sizes = [3.5,3.5,3.5,3.5,3.5,3.5,3.5,3.5,3.5,3.5,3.5,3.5,3.5]
    select_plots = []
    for p in resfile.dct.keys():
        select_plots.append(p)
    numplots = len(select_plots)
    plotrows = int(numplots/args.y)
    if args.y > 1:
        plotextras = numplots%args.y
    else:
        plotextras = 0
    extraflag=False
    if plotextras > 0:
        plotrows += 1
        extraflag=True
    plotno=0
    plotrow=0
    print('numplots:',numplots)
    print('plotrows:',plotrows)
    print('plotextras:',plotextras)
    fig = plt.figure(figsize=(args.y*10*0.7, 4*plotrows*0.7), dpi= 80,)
    for k, v in resfile.dct.items():
        desc_label = k
        ax = fig.add_subplot(plotrows, args.y, plotno+1)
        current_lines = []
        if (plotno+1)%args.y==1 or args.y==1:
            plotrow += 1
        res = v
        ax.plot((1.0E-14,1.0), (1.0E-16, 1.0E-16), 'k-.', lw=2)
        ci = 0
        for ri, r in enumerate(res.keys()):
            addthis=True
            feature_num = res[r]['feature_num']
            xdata = res[r]['percent']
            ydata = res[r]['total']
            ldata = r
            if addthis:
                if 'ACE (' in desc_label:
                    if ri<1:
                        color_codes = plt.cm.Paired(np.arange(0,numplots*2)) 
                        color_maps = [1,18,16,0,8,0,3]
                        colorid = color_maps[int(ci)]
                    else:
                        color_codes = ['orange','mediumblue','dodgerblue','deepskyblue']
                        color_maps = [0,1,2,3,4,0,3]
                        colorid = color_maps[int(ci)]
                elif 'MBTR' in desc_label:
                    if ri>3:
                        colorid = color_maps[int(ci)]
                    else:
                        colorid = color_maps[int(ci)]
                else:
                    colorid = color_maps[int(ci)]
                thisplot, = ax.plot(xdata, ydata, color=color_codes[colorid], lw=line_sizes[int(ci)], linestyle=line_types[int(ci)], label=ldata+" ("+str(feature_num)+")")
                current_lines.append(thisplot)
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
            ax.set_ylabel(r'$||V_{all}-V_{sel}||$', fontsize=14, fontproperties=fontp)
            plt.tick_params(
                axis='both',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                left=True,      # ticks along the bottom edge are off
                right=False,         # ticks along the top edge are off
                labelleft=True,
                bottom=True,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
        if extraflag is False and plotrow==plotrows:
            ax.set_xlabel(r'$\mathrm{Percentage\,\,of\,\,Features\,}(\%)$', fontproperties=fontp)
            plt.tick_params(
                axis='both',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=True,      # ticks along the bottom edge are off
                left=True,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=True) # labels along the bottom edge are off
        if extraflag and plotextras!=0 and plotrow==plotrows-1 and (plotno+1)%args.y not in [i+1 for i in range(plotextras)]:
            ax.set_xlabel(r'$\mathrm{Percentage\,\,of\,\,Features\,}(\%)$', fontproperties=fontp)
            plt.tick_params(
                axis='both',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=True,      # ticks along the bottom edge are off
                left=True,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=True) # labels along the bottom edge are off
        if extraflag and plotextras!=0 and plotrow==plotrows and (plotno+1)%args.y in [i+1 for i in range(plotextras)]:
            ax.set_xlabel(r'$\mathrm{Percentage\,\,of\,\,Features\,}(\%)$', fontproperties=fontp)
            plt.tick_params(
                axis='both',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=True,      # ticks along the bottom edge are off
                left=True,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=True) # labels along the bottom edge are off
        ax.set_xticks(range(0,101,5))
        ax.set_xticklabels(ax.get_xticks(), fontx)
        plt.xlim((-1,101))
        plt.ylim((1.0E-12,1.0))
        if 'MBTR' in desc_label:
            first_legend = ax.legend(handles=[ l for l in current_lines[:3]],
                                     fontsize=14, loc=(0.01,0.00), framealpha=0)
            ax1 = plt.gca().add_artist(first_legend)
            plt.legend(handles=[l for l in current_lines[3:]],
                       fontsize=14, loc=(0.4,0.00), framealpha=0)
        elif 'ACE (AlNiCu)' in desc_label:
            ax.legend(fontsize=14, loc=(0.01,0.00), framealpha=0)
        else:
            if plotno == 0:
                ax.legend(fontsize=14, framealpha=0)
            else:
                ax.legend(fontsize=14, framealpha=0, loc=(0.01,0.00))
        addtext=True
        if addtext:
            plt.text(0.5-0.021*len(desc_label)/2, 0.9, desc_label, fontdict=font, transform=ax.transAxes)
            if (plotno+1)%args.y!=1 and args.y>1:
                plt.text(-0.05, 0.9, chr(plotno+97)+')', fontdict=font, transform=ax.transAxes)
            if (plotno+1)%args.y==1 or args.y==1:
                plt.text(-0.12, 0.9, chr(plotno+97)+')', fontdict=font, transform=ax.transAxes)
        plotno+=1
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    #if args.outfile is not None:
    #    base = os.path.basename(args.outfile)
    #else:
    #    base = os.path.basename(args.filename)
    #filebase = os.path.splitext(base)[0]
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.03, hspace=0.08)
    fig.savefig('figure6.pdf',dpi= 100)
    print('Figure6.pdf generated.')
    
