import os, os.path as osp, json, argparse, sys, fnmatch, json
import numpy as np

import matplotlib.pyplot as plt


sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import trigger as trig
from build_histogram import HistogramCollection


trig.set_mpl_fontsize()

BKGS = ['qcd', 'ttjets', 'wjets', 'zjets']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('histograms_file', type=str)
    parser.add_argument('--var', type=str, default='pt', choices=['pt', 'pt_subl', 'ht', 'met'])
    parser.add_argument('--bkg', type=str, nargs='*', default=['all'], choices=BKGS + ['comb', 'all'])
    parser.add_argument('--nomet', action='store_true')
    parser.add_argument('--fit', action='store_true')
    args = parser.parse_args()

    collection = HistogramCollection.load(args.histograms_file)

    fig = plt.figure(figsize=(8,8))
    ax = fig.gca()

    if args.fit:
        bkgs = ['comb']
    elif 'all' in args.bkg:
        bkgs = BKGS + ['comb']
    else:
        bkgs = args.bkg

    # Create the 'combined' histograms (sum of all bkgs);
    # Store in the collection as if it's a normal bkg
    if 'comb' in bkgs:
        keys = list(collection.hists.keys())
        for thing in ['nomettrig', 'inclmettrig', 'notrig']:
            collection.hists[f'{args.var}_comb_{thing}'] = sum(
                collection.hists[k] for k in fnmatch.filter(keys, f'{args.var}_*_{thing}')
                )

    for bkg in bkgs:
        h_nomettrig = collection.hists[args.var+'_'+bkg+'_nomettrig']
        h_inclmettrig = collection.hists[args.var+'_'+bkg+'_inclmettrig']
        h = h_nomettrig if args.nomet else h_inclmettrig

        h_notrig = collection.hists[args.var+'_'+bkg+'_notrig']

        eff = h.counts / h_notrig.counts
        bin_centers = .5*(h_notrig.binning[:-1] + h_notrig.binning[1:])

        line = ax.plot(bin_centers, eff, 'o')[0]

        if args.fit:
            def sigmoid(x):
                return 1./(1+np.exp(-x))
            def inv_sigmoid(y):
                return -np.log(1./y-1)
            # First transform y with inv sigmoid; resulting points
            # should be suitable for poly1d
            y_tr = inv_sigmoid(eff)
            # High degree polynomial to capture the curve in detail
            res = np.polyfit(bin_centers, y_tr, 20)
            fit = np.poly1d(res)
            x_fine = np.linspace(h_notrig.binning[1], h_notrig.binning[-1], 100)
            ax.plot(x_fine, sigmoid(fit(x_fine)), '-', color=line.get_color(), label='fit')
            with open('bkg_trigeff_fit_2018.txt', 'w') as f:
                json.dump(list(res), f)

        else:
            interpolation = trig.Interpolation(bin_centers, eff)
            print(f'Eff for {bkg} at 500 GeV: {interpolation(500.)}')
            ax.plot(
                *interpolation.fine(), c=line.get_color(),
                label=f'{bkg} 98\% = {interpolation.solve(.98):.1f} GeV'
                )

    ax.legend(fontsize=16)

    ax.text(
        .0, 1.0,
        r'\textbf{CMS}\,\fontsize{19pt}{3em}\selectfont{}{\textit{Simulation Preliminary}}',
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax.transAxes,
        usetex=True,
        fontsize=23
        )
    ax.text(
        1.0, 1.0,
        r'2018 (13 TeV)',
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax.transAxes,
        usetex=True,
        fontsize=19
        )

    var_title = {
        'pt' : 'Leading AK8 $\mathrm{p}_\mathrm{T}$ (GeV)',
        'pt_subl' : 'Subleading AK8 $\mathrm{p}_\mathrm{T}$ (GeV)',
        'ht' : 'HT (GeV)',
        'met' : 'MET (GeV)',
        }[args.var]

    ax.set_xlabel(var_title)
    ax.set_ylabel('Efficiency')

    outfile = 'bkgeff.png'
    plt.savefig(outfile, bbox_inches='tight')
    os.system(f'imgcat {outfile}')



if __name__ == '__main__':
    main()
