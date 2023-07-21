import os, os.path as osp, glob, copy, argparse, sys, uuid, logging, logging.config, re, shutil, pprint
from contextlib import contextmanager

import numpy as np

import svj_ntuple_processing as svj


def setup_logger(name: str = "triggerstudy") -> logging.Logger:
    """Sets up a Logger instance.

    If a logger with `name` already exists, returns the existing logger.

    Args:
        name (str, optional): Name of the logger. Defaults to "demognn".

    Returns:
        logging.Logger: Logger object.
    """
    if name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.info("Logger %s is already defined", name)
    else:
        fmt = logging.Formatter(
            fmt=(
                "\033[34m[%(name)s:%(levelname)s:%(asctime)s:%(module)s:%(lineno)s]\033[0m"
                + " %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
    return logger


logger = setup_logger()


def debug(flag: bool = True) -> None:
    """Convenience switch to set the logging level to DEBUG.

    Args:
        flag (bool, optional): If true, set the logging level to DEBUG. Otherwise, set
            it to INFO. Defaults to True.
    """
    if flag:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)


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


def filter_triggers(cols, incl_met=True, triggers=svj.triggers_2018):
    if isinstance(triggers, int):
        triggers = {
            2018 : svj.triggers_2018,
            2017 : svj.triggers_2017,
            2016 : svj.triggers_2016,
            }[triggers]
    if not incl_met: triggers = [t for t in triggers if 'MET' not in t]
    indices = np.array([cols.metadata['trigger_titles'].index(t) for t in triggers])
    return np.any(cols.arrays['triggers'][:,indices] == 1, axis=-1)


def put_on_cmslabel(ax, text='Simulation Preliminary', year=2018):
    ax.text(
        .0, 1.0,
        r'\textbf{CMS}\,\fontsize{19pt}{3em}\selectfont{}{\textit{'+text+'}}',
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax.transAxes,
        usetex=True,
        fontsize=23
        )
    ax.text(
        1.0, 1.0,
        '{} (13 TeV)'.format(year),
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax.transAxes,
        usetex=True,
        fontsize=19
        )

var_titles = {
    'pt' : 'Leading AK8 $\mathrm{p}_\mathrm{T}$ (GeV)',
    'pt_subl' : 'Subleading AK8 $\mathrm{p}_\mathrm{T}$ (GeV)',
    'pt_ak15' : 'Leading AK15 $\mathrm{p}_\mathrm{T}$ (GeV)',
    'pt_ak15_subl' : 'Subleading AK15 $\mathrm{p}_\mathrm{T}$ (GeV)',
    'ht' : 'HT (GeV)',
    'met' : 'MET (GeV)',
    'mt_ak15_subl' : '$\mathrm{M}_\mathrm{T}$ of subleading AK15 jet (GeV)'
    }

# Define binning per variable
binning = dict(
    pt = np.linspace(0., 800., 50),
    pt_subl = np.linspace(0., 800., 50),
    pt_ak15 = np.linspace(0., 800., 50),
    pt_ak15_subl = np.linspace(0., 800., 50),
    ht = np.linspace(0., 1400., 60),
    met = np.linspace(0., 700., 50),
    mt_ak15_subl = np.linspace(0., 1000., 40),
    )

variables = list(sorted(binning.keys()))




def pull_arg(*args, **kwargs) -> argparse.Namespace:
    """Reads a specific argument out of sys.argv, and then deletes that argument from
    sys.argv.

    This useful to build very adaptive command line options to scripts. It does
    sacrifice auto documentation of the command line options though.

    Returns:
        argparse.Namespace: Namespace object for only the specific argument.
    """

    """
    Reads a specific argument out of sys.argv, and then
    deletes that argument from sys.argv.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(*args, **kwargs)
    args, other_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + other_args
    return args


@contextmanager
def timeit(msg):
    """
    Prints duration of a block of code in the terminal.
    """
    try:
        logger.info(msg)
        sys.stdout.flush()
        t0 = time.time()
        yield None
    finally:
        t1 = time.time()
        logger.info(f"Done {msg[0].lower() + msg[1:]}, took {t1-t0:.2f} secs")


class Scripter:
    """
    Command line utility.

    When an instance of this class is used as a contextwrapper on a function, that
    function will be considered a 'command'.

    When Scripter.run() is called, the script name is pulled from the command line, and
    the corresponding function is executed.

    Example:

        In file test.py:
        >>> scripter = Scripter()
        >>> @scripter
        >>> def my_func():
        >>>     print('Hello world!')
        >>> scripter.run()

        On the command line, the following would print 'Hello world!':
        $ python test.py my_func
    """

    def __init__(self):
        self.scripts = {}

    def __call__(self, fn):
        """
        Stores a command line script with its name as the key.
        """
        self.scripts[fn.__name__] = fn
        return fn

    def run(self):
        script = pull_arg("script", choices=list(self.scripts.keys())).script
        logger.info(f"Running {script}")
        self.scripts[script]()


@contextmanager
def quick_ax(figsize=(10, 10), outfile="tmp.png"):
    """
    Context manager to open a matplotlib Axes.
    Upon closing, saves it to a file and calls
    imgcat (an iTerm2 command line util) on it
    to display the plot in the terminal.
    """
    try:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        yield ax
    finally:
        plt.savefig(outfile, bbox_inches="tight")
        try:
            os.system(f"imgcat {outfile}")
        except Exception:
            pass


@contextmanager
def quick_fig(figsize=(10, 10), outfile="tmp.png"):
    """
    Context manager to open a matplotlib Figure.
    Upon closing, saves it to a file and calls
    imgcat (an iTerm2 command line util) on it
    to display the plot in the terminal.
    """
    try:
        fig = plt.figure(figsize=figsize)
        yield fig
    finally:
        plt.savefig(outfile, bbox_inches="tight")
        try:
            os.system(f"imgcat {outfile}")
        except Exception:
            pass
