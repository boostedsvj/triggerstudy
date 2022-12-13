import os, os.path as osp, json, argparse, sys, fnmatch, json
import numpy as np

import matplotlib.pyplot as plt

import svj_ntuple_processing as svj

import trigger as trig



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('npzfiles', type=str, nargs='+')
    parser.add_argument('--var', type=str, default='pt', choices=['pt', 'pt_subl', 'ht', 'met'])
    parser.add_argument('--nomet', action='store_true')
    args = parser.parse_args()

    fig = plt.figure(figsize=(8,8))
    ax = fig.gca()

    # binning = trig.binning[args.var]
    binning = np.linspace(0., 630., 30)

    bin_centers = .5*(binning[1:] + binning[:-1])

    for col in [ svj.Columns.load(f) for f in args.npzfiles ]:
        passes = trig.filter_triggers(col, not(args.nomet))

        vals = col.arrays[args.var]

        revcumsum = lambda x: np.cumsum(x[::-1])[::-1]
        h_notrig = revcumsum(np.histogram(vals, binning)[0])
        h_trig = revcumsum(np.histogram(vals[passes], binning)[0])

        

        eff = np.where(h_notrig!=0., h_trig/h_notrig, 0.)

        meta = col.metadata
        label = f"mz{meta['mz']}, rinv{meta['rinv']}"
        line = ax.plot(bin_centers, eff, 'o')[0]

        try:
            interpolation = trig.Interpolation(bin_centers, eff)
            print(f'Eff for {label} at 500 GeV: {interpolation(500.)}')
            ax.plot(
                *interpolation.fine(), c=line.get_color(),
                label=f'{label} 98\% = {interpolation.solve(.98):.1f} GeV'
                )
        except Exception:
            svj.logger.error(f'Interpolation failed for {label}')

    ax.legend(fontsize=16)
    trig.put_on_cmslabel(ax)
    ax.set_xlabel(trig.var_titles[args.var])
    ax.set_ylabel('Efficiency')

    outfile = 'sigeff.png'
    plt.savefig(outfile, bbox_inches='tight')
    os.system(f'imgcat {outfile}')


if __name__ == '__main__':
    main()