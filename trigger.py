import os, os.path as osp, glob, copy, argparse, sys, uuid, logging, logging.config, re, shutil, pprint

import numpy as np

import svj_ntuple_processing as svj


logging.config.dictConfig({ 
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': { 
        'standard': { 
            'format': '\033[94m%(name)s:%(asctime)s:%(levelname)s:%(module)s:%(funcName)s:%(lineno)s:\033[0m %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
            },
        },
    'handlers': { 
        'default': { 
            'level': 'INFO', 'formatter': 'standard', 'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
            },
        },
    'loggers': { 
        'triggerstudy': {'handlers': ['default'], 'level': 'DEBUG', 'propagate': False},
        } 
    })
logger = logging.getLogger('triggerstudy')



cms_style = {
    "font.sans-serif": ["TeX Gyre Heros", "Helvetica", "Arial"],
    "font.family": "sans-serif",
    # 
    "mathtext.fontset": "custom",
    "mathtext.rm": "helvetica",
    "mathtext.bf": "helvetica:bold",
    "mathtext.sf": "helvetica",
    "mathtext.it": "helvetica:italic",
    "mathtext.tt": "helvetica",
    "mathtext.cal": "helvetica",
    # 
    "figure.figsize": (10.0, 10.0),
    "font.size": 26,
    "axes.labelsize": "medium",
    "axes.unicode_minus": False,
    "xtick.labelsize": "small",
    "ytick.labelsize": "small",
    "legend.fontsize": "small",
    "legend.handlelength": 1.5,
    "legend.borderpad": 0.5,
    "legend.frameon": False,
    "xtick.direction": "in",
    "xtick.major.size": 12,
    "xtick.minor.size": 6,
    "xtick.major.pad": 6,
    "xtick.top": True,
    "xtick.major.top": True,
    "xtick.major.bottom": True,
    "xtick.minor.top": True,
    "xtick.minor.bottom": True,
    "xtick.minor.visible": True,
    "ytick.direction": "in",
    "ytick.major.size": 12,
    "ytick.minor.size": 6.0,
    "ytick.right": True,
    "ytick.major.left": True,
    "ytick.major.right": True,
    "ytick.minor.left": True,
    "ytick.minor.right": True,
    "ytick.minor.visible": True,
    "grid.alpha": 0.8,
    "grid.linestyle": ":",
    "axes.linewidth": 2,
    "savefig.transparent": False,
    "xaxis.labellocation": "right",
    "yaxis.labellocation": "top",
    'text.usetex' : True,    
}

def set_mpl_fontsize(small=16, medium=22, large=26):
    plt.rc('font', size=small)          # controls default text sizes
    plt.rc('axes', titlesize=small)     # fontsize of the axes title
    plt.rc('axes', labelsize=medium)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=small)    # fontsize of the tick labels
    plt.rc('legend', fontsize=medium)    # legend fontsize
    plt.rc('figure', titlesize=large)  # fontsize of the figure title
    from matplotlib.pyplot import style as plt_style
    plt_style.use(cms_style)
    plt.rc('text', usetex=True)
    plt.rc(
        'text.latex',
        preamble=(
            r'\usepackage{helvet} '
            r'\usepackage{sansmath} '
            r'\sansmath '
            )    
        )

try:
    import matplotlib.pyplot as plt
    set_mpl_fontsize()
except ImportError:
    logger.warning('Could not import matplotlib')


class Interpolation:

    def __init__(self, x, y):
        from scipy.interpolate import interp1d
        self.x = x
        self.y = y
        self.f_lin = interp1d(x, y)

    def __call__(self, x):
        return self.f_lin(x)

    def solve(self, y_val):
        i_init = np.argmin(np.abs(self.y-y_val))
        from scipy.optimize import fsolve
        x = fsolve(lambda x: self.f_lin(x) - y_val, self.x[i_init])[0]
        return x

    def fine(self, n=500):
        x = np.linspace(self.x[0], self.x[-1], n)
        return x, self(x)


def filter_triggers(cols, incl_met=True):
    triggers = svj.triggers_2018
    if not incl_met: triggers = [t for t in triggers if 'MET' not in t]
    indices = np.array([cols.metadata['trigger_titles'].index(t) for t in triggers])
    return np.any(cols.arrays['triggers'][:,indices], axis=-1)


def put_on_cmslabel(ax):
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

var_titles = {
    'pt' : 'Leading AK8 $\mathrm{p}_\mathrm{T}$ (GeV)',
    'pt_subl' : 'Subleading AK8 $\mathrm{p}_\mathrm{T}$ (GeV)',
    'ht' : 'HT (GeV)',
    'met' : 'MET (GeV)',
    }

# Define binning per variable
binning = dict(
    pt = np.linspace(0., 800., 50),
    pt_subl = np.linspace(0., 800., 50),
    ht = np.linspace(0., 1400., 60),
    met = np.linspace(0., 700., 50),
    )
