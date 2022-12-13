import os, os.path as osp, multiprocessing as mp, json, argparse, itertools, traceback, re, sys
from time import strftime

import tqdm
import numpy as np
import seutils
seutils.MAX_RECURSION_DEPTH = 1000

import svj_ntuple_processing as svj
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import trigger as trig


class Histogram:
    @classmethod
    def fromdict(cls, dct):
        inst = cls(np.array(dct['binning']))
        inst.counts = np.array(dct['counts'])
        inst.counts_w2 = np.array(dct['counts_w2'])
        return inst

    def __init__(self, binning):
        self.binning = binning
        self.counts = np.zeros(self.nbins)
        self.counts_w2 = np.zeros(self.nbins)

    @property
    def nbins(self):
        return len(self.binning)-1

    def fill(self, values, weights):
        self.counts += np.histogram(values, self.binning, weights=weights)[0]
        self.counts_w2 += np.histogram(values, self.binning, weights=weights**2)[0]

    def json(self):
        """
        Converts histogram to a JSON-saveable dict
        """
        return {
            'binning' : list(self.binning),
            'counts' : list(self.counts),
            'counts_w2' : list(self.counts_w2),
            }

    def __repr__(self):
        return f'<Histogram {self.nbins} bins, {self.counts.sum():.3e} entries>'

    def __add__(self, o):
        np.testing.assert_array_equal(self.binning, o.binning)
        out = Histogram(self.binning[:])
        out.counts = self.counts + o.counts
        out.counts_w2 = self.counts_w2 + o.counts_w2
        return out

    def __radd__(self, o):
        if o == 0: return self
        raise NotImplemented


class HistogramCollection:

    @classmethod
    def load(cls, infile):
        with open(infile, 'r') as f:
            d = json.load(f)
        inst = cls()
        inst.hists = {name : Histogram.fromdict(histdct) for name, histdct in d['hists'].items()}
        inst.files = set(d['files'])
        inst.infile = infile
        return inst

    def __init__(self):
        self.hists = {}
        self.files = set()

    def json(self):
        out = {}
        out['files'] = list(self.files)
        out['hists'] = {name : hist.json() for name, hist in self.hists.items()}
        return out

    def save(self, outfile):
        os.makedirs(osp.dirname(osp.abspath(outfile)), exist_ok=True)
        svj.logger.debug('Dumping hists to %s', outfile)
        with open(outfile, 'w') as f:
            json.dump(self.json(), f)

    def ls(self):
        print(getattr(self, 'infile', '<in memory>'))
        for name, hist in self.hists.items():
            print(f'  {name}: {hist}')


def get_npzfiles(pat, cache_file='cache_npzfiles.json'):
    if osp.isfile(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
    else:
        cache = {}
        
    if pat in cache:
        svj.logger.info(f'Returning npzfiles for pat {pat} from cache')
        return cache[pat]

    svj.logger.info('Building list of npzfiles to include')
    cache[pat] = seutils.ls_wildcard(pat)

    svj.logger.info(f'Dumping cache in {cache_file}')
    with open(cache_file, 'w') as f:
        json.dump(cache, f)


def worker(tup):
    npzfile, lock, histogram_file, variables = tup

    try:
        cols = svj.Columns.load(npzfile)
        passes_incl_met = trig.filter_triggers(cols, True)
        passes_no_met = trig.filter_triggers(cols, False)
        bkg = cols.metadata['bkg']
        
        with lock:
            histograms = HistogramCollection.load(histogram_file)

            if npzfile in histograms.files:
                svj.logger.info(f'File {npzfile} already present in the collection, skipping.')
                return
            histograms.files.add(npzfile)

            for v in variables:
                vals = cols.arrays[v]
                weights = cols.arrays['weight']
                histograms.hists[v+'_'+bkg+'_notrig'].fill(vals, weights)
                histograms.hists[v+'_'+bkg+'_inclmettrig'].fill(vals[passes_incl_met], weights[passes_incl_met])
                histograms.hists[v+'_'+bkg+'_nomettrig'].fill(vals[passes_no_met], weights[passes_no_met])

            histograms.save(histogram_file)
    except Exception:
        svj.logger.error(f'Problem processing {npzfile}:\n{traceback.format_exc()}\n{cols.metadata}')
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('npzfiles', type=str, nargs='+')
    args = parser.parse_args()

    npzfiles = list(itertools.chain(*(get_npzfiles(pat) for pat in args.npzfiles)))

    # Filter out QCD with pt<170
    pt_left = lambda f: int(re.search(r'Pt_(\d+)to', f).group(1))
    npzfiles = [f for f in npzfiles if ('QCD' not in f or pt_left(f) >= 170)]

    histograms_file = strftime('histograms_%b%d.json')
    variables = ['pt', 'ht', 'met', 'pt_subl']
    bkgs = ['qcd', 'ttjets', 'wjets', 'zjets']

    if not osp.isfile(histograms_file):
        histograms = HistogramCollection()
    
        # Define the histograms
        for v in variables:
            for b in bkgs:
                histograms.hists[v+'_'+b+'_notrig'] = Histogram(trig.binning[v])
                histograms.hists[v+'_'+b+'_inclmettrig'] = Histogram(trig.binning[v])
                histograms.hists[v+'_'+b+'_nomettrig'] = Histogram(trig.binning[v])

        # Save empty histograms to a file
        histograms.save(histograms_file)

    # Launch the workers
    with mp.Manager() as manager:
        lock = manager.Lock()
        mp_args = [(npz, lock, histograms_file, variables) for npz in npzfiles]
        with mp.Pool(16) as p:
            r = list(tqdm.tqdm(p.imap(worker, mp_args), total=len(mp_args)))

if __name__ == '__main__':
    main()