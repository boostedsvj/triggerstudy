from contextlib import contextmanager
import os, os.path as osp, glob, copy, argparse, sys, uuid, logging, logging.config, re, shutil, pprint
from time import strftime

import numpy as np, uproot
import awkward as ak
import seutils
import matplotlib.pyplot as plt

import svj_ntuple_processing as svj

import trigger as trig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('npzs', nargs='+', type=str)
    parser.add_argument('-o', '--outfile', type=str, default='jetptdist.png')
    parser.add_argument('--highptzoomin', action='store_true')
    args = parser.parse_args()

    fig = plt.figure(figsize=(10,10))
    ax = fig.gca()
    jetpt_axis = np.linspace(0., 600., 40)
    if args.highptzoomin: jetpt_axis = np.linspace(500., 800., 40)

    cols = [ svj.Columns.load(npz) for npz in args.npzs ]
    cols.sort(key=lambda h: h.metadata['madpt'])

    for col in cols:
        ax.hist(
            col.arrays['pt'], jetpt_axis,
            label=f"madpt cut: {col.metadata['madpt']} GeV",
            histtype=u'step',
            density=bool(args.highptzoomin)
            )

    ax.legend(fontsize=16)
    ax.set_xlabel(r'$p_{T}^{jet}$ (GeV)')
    ax.set_ylabel(r'Count')

    outdir = osp.dirname(osp.abspath(args.outfile))
    if not osp.isdir(outdir): os.makedirs(outdir)
    plt.savefig(args.outfile, bbox_inches='tight')
    os.system('imgcat ' + args.outfile)


if __name__ == '__main__': main()