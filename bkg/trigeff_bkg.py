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
    parser.add_argument('--var', type=str, default='pt', choices=trig.variables)
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

        
        revcumsum = lambda x: np.cumsum(x[::-1])[::-1]
        eff = revcumsum(h.counts) / revcumsum(h_notrig.counts)

        if args.fit:
            print('Warning: using np.maximum.accumulate on efficiency')
            eff = np.maximum.accumulate(eff)

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
            res = np.polyfit(bin_centers, y_tr, 15)
            fit = np.poly1d(res)
            x_fine = np.linspace(h_notrig.binning[1], h_notrig.binning[-1], 100)
            ax.plot(x_fine, sigmoid(fit(x_fine)), '-', color=line.get_color(), label='fit')
            outfile = 'bkg_trigeff_fit_2018.txt'
            print('Dumping fit result to {}'.format(outfile))
            with open(outfile, 'w') as f:
                json.dump(list(res), f)
        else:
            interpolation = trig.Interpolation(bin_centers, eff)
            print(f'Eff for {bkg} at 500 GeV: {interpolation(500.)}')
            ax.plot(
                *interpolation.fine(), c=line.get_color(),
                label=f'{bkg} 98\% = {interpolation.solve(.98):.1f} GeV'
                )

    ax.legend(fontsize=16)
    trig.put_on_cmslabel(ax)
    ax.set_xlabel(trig.var_titles[args.var])
    ax.set_ylabel('Efficiency')

    outfile = 'bkgeff.png'
    plt.savefig(outfile, bbox_inches='tight')
    os.system(f'imgcat {outfile}')



if __name__ == '__main__':
    main()
