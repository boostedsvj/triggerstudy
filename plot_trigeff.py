import os, os.path as osp, json, argparse, sys, fnmatch, json
import numpy as np
import tqdm
import scipy.stats

import seutils

import matplotlib.pyplot as plt

import svj_ntuple_processing as svj

import trigger as trig


scripter = trig.Scripter()


def reverse_cumsum(x):
    """
    Cumsum from right to left
    """
    return np.cumsum(x[::-1])[::-1]


# trig.logger.warning('REMOVING SOME 2016 TITLES FOR NOW')
# svj.triggers_2016.remove('HLT_PFHT800_v')
# svj.triggers_2016.remove('HLT_PFHT300_PFMET100_v')
# svj.triggers_2016.remove('HLT_PFHT300_PFMET110_v')

# svj.triggers_2016.extend([
#     'HLT_PFMET120_PFMHT120_IDTight_v',
#     'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v',
#     ])
# svj.triggers_2018.extend([
#     'HLT_PFMET120_PFMHT120_IDTight_v',
#     'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v',
#     ])

svj.triggers_2017.append('HLT_PFMETTypeOne120_PFMHT120_IDTight_v')
svj.triggers_2018.extend([
    'HLT_PFMET120_PFMHT120_IDTight_v',
    'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v',
    ])


def poisson_err_up(n):
    """
    Python reimplementation of PoissonErrorUp in 
    https://github.com/kpedro88/Analysis/blob/SVJ2018/KCode/KMath.h
    """
    alpha = 1 - 0.682689492
    # double U = (ROOT::Math::gamma_quantile_c(alpha/2,N+1,1.));
    # >>> ROOT.Math.gamma_quantile_c(alpha/2., 11, 1.)
    # 14.266949759891316
    # >>> scipy.stats.gamma.ppf(1.-alpha/2., 11)
    # 14.266949759891313
    u = scipy.stats.gamma.ppf(1-alpha/2., n+1)
    return u-n


def calc_eff_error(passes, total):
    """
    Python reimplementation of EffError in 
    https://github.com/kpedro88/Analysis/blob/SVJ2018/KCode/KMath.h
    """
    n = len(passes)
    nonzero = total != 0.

    passes = passes[nonzero]
    total = total[nonzero]
    fail = total - passes

    eff = passes / total

    err_pass = poisson_err_up(passes)
    err_fail = poisson_err_up(fail)

    # return 
    #   sqrt(pow(1-eff,2)*pow(err_p,2)
    #   + pow(eff,2)*pow(err_f,2)) / (double)total;
    err_eff = np.sqrt(
        (1.-eff)**2 * err_pass**2
        + eff**2 * err_fail**2
        ) / total

    # Limit eff + err < 1., eff - err > 0.
    err_eff_up = np.where(eff + err_eff <= 1., err_eff, 1.-eff)
    err_eff_down = np.where(eff - err_eff >= 0., err_eff, eff)

    # Fill in calculated err wherever the total was nonzero
    err_eff_up_final = np.zeros(n)
    err_eff_up_final[nonzero] = err_eff_up

    err_eff_down_final = np.zeros(n)
    err_eff_down_final[nonzero] = err_eff_down

    return np.row_stack((err_eff_down_final, err_eff_up_final))



@scripter
def singlemuon():
    parser = argparse.ArgumentParser()
    parser.add_argument('npzfiles', type=str, nargs='+')
    parser.add_argument('--var', type=str, default='pt', choices=trig.variables)
    parser.add_argument('-y', '--year', type=int, default=2018)
    args = parser.parse_args()

    # binning = trig.binning[args.var]
    binning = np.linspace(0., 1200., 60)

    if args.year == 2016:
        binning = np.concatenate((
            np.linspace(0., 800., 39, endpoint=False),
            np.linspace(820., 1200., 10), 
            ))

    bin_centers = .5*(binning[1:] + binning[:-1])

    ax2_y = 0.

    npzfiles = []
    for f in args.npzfiles:
        if seutils.path.has_protocol(f) and '*' in f:
            npzfiles.extend(seutils.ls_wildcard(f))
        else:
            npzfiles.append(f)

    cols = []
    for f in tqdm.tqdm(npzfiles):
        cols.append(svj.Columns.load(f, encoding='latin1'))
    col = svj.concat_columns(cols)
    del cols

    with trig.quick_ax() as ax:

        passes_singlemuon = trig.filter_triggers(col, triggers=['HLT_Mu50_v'])
        passes = trig.filter_triggers(col, triggers=args.year) & passes_singlemuon

        vals = col.arrays[args.var]

        h_singlemuon = np.histogram(vals[passes_singlemuon], binning)[0]
        h_trig = np.histogram(vals[(passes)], binning)[0]
        ax2_y = max(ax2_y, np.max(h_trig))

        # numerator = reverse_cumsum(h_trig)
        # denominator = reverse_cumsum(h_singlemuon)
        numerator = h_trig
        denominator = h_singlemuon
        eff = np.where(denominator!=0., numerator/denominator, 0.)

        err_eff = calc_eff_error(numerator, denominator)


        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
        ax.errorbar(bin_centers, eff, yerr=err_eff, fmt='o', color=color)

        # color = ax.plot(bin_centers, eff, 'o')[0].get_color()

        try:
            interpolation = trig.Interpolation(bin_centers, eff)
            ax.plot(
                *interpolation.fine(), c=color,
                label=f'95\% = {interpolation.solve(.95):.1f} GeV'
                )
        except Exception:
            svj.logger.error(f'Interpolation failed')

        ax2 = ax.twinx()
        ax2.hist(vals[passes], binning, color=color, alpha=.5)
        ax2.step(binning[:-1], h_singlemuon, '--', c=color, where='post')
        
        ax.set_ylim(0, 1.15)
        ax2.set_ylim(0., 1.5*ax2_y)

        ax.legend(fontsize=16)
        trig.put_on_cmslabel(ax, text='Preliminary', year=args.year)
        ax.set_xlabel(trig.var_titles[args.var])
        ax.set_ylabel('Efficiency')




@scripter
def mc():
    parser = argparse.ArgumentParser()
    parser.add_argument('npzfiles', type=str, nargs='+')
    parser.add_argument('--var', type=str, default='pt', choices=trig.variables)
    parser.add_argument('--nomet', action='store_true')
    args = parser.parse_args()

    fig = plt.figure(figsize=(8,8))
    ax = fig.gca()

    binning = trig.binning[args.var]
    bin_centers = .5*(binning[1:] + binning[:-1])


    ax2_y = 0.

    with trig.quick_ax() as ax:
        ax2 = ax.twinx()

        for col in [ svj.Columns.load(f) for f in args.npzfiles ]:
            passes = trig.filter_triggers(col, not(args.nomet))
            vals = col.arrays[args.var]


            h_all = np.histogram(vals, binning)[0]
            h_trig = np.histogram(vals[passes], binning)[0]

            ax2_y = max(ax2_y, np.max(h_trig))

            # numerator = reverse_cumsum(h_trig)
            # denominator = reverse_cumsum(h_all)
            numerator = h_trig
            denominator = h_all
            eff = np.where(denominator!=0., numerator/denominator, 0.)

            meta = col.metadata

            try:
                label = f"mz{meta['mz']}, rinv{meta['rinv']}"
            except KeyError:
                label = osp.basename(col.metadata['src']).replace('.npz','')

            color = ax.plot(bin_centers, eff, 'o')[0].get_color()

            ax2.hist(vals[passes], binning, color=color, alpha=.5)
            ax2.step(binning[:-1], h_all, '--', c=color, where='post')

            try:
                interpolation = trig.Interpolation(bin_centers, eff)
                print(f'Eff for {label} at 500 GeV: {interpolation(500.)}')
                ax.plot(
                    *interpolation.fine(), c=color,
                    label=f'{label} 98\% = {interpolation.solve(.98):.1f} GeV'
                    )
            except Exception:
                svj.logger.error(f'Interpolation failed for {label}')

        ax2.set_ylim(0., 1.5*ax2_y)
        ax.legend(fontsize=16)
        trig.put_on_cmslabel(ax)
        ax.set_xlabel(trig.var_titles[args.var])
        ax.set_ylabel('Efficiency')


if __name__ == '__main__':
    scripter.run()
