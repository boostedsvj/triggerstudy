import os, os.path as osp, glob, copy, argparse, sys, uuid, logging, logging.config, re, shutil, pprint

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


def set_mpl_fontsize(small=16, medium=22, large=26):
    plt.rc('font', size=small)          # controls default text sizes
    plt.rc('axes', titlesize=small)     # fontsize of the axes title
    plt.rc('axes', labelsize=medium)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=small)    # fontsize of the tick labels
    plt.rc('legend', fontsize=medium)    # legend fontsize
    plt.rc('figure', titlesize=large)  # fontsize of the figure title

try:
    import matplotlib.pyplot as plt
    set_mpl_fontsize()
except ImportError:
    logger.warning('Could not import matplotlib')
