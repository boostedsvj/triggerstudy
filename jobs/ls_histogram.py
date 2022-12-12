import json, argparse
import numpy as np

from build_histogram import HistogramCollection

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('histograms_file', type=str)
    args = parser.parse_args()

    histograms = HistogramCollection.load(args.histograms_file)

    for name, hist in histograms.hists.items():
        print(name)
        print(f'  {hist.binning}')
        print(f'  {hist.counts}')
        print(f'  {np.sqrt(hist.counts_w2)}')






if __name__ == '__main__':
    main()
