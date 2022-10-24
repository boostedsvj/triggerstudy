from contextlib import contextmanager
import os, os.path as osp, glob, copy, argparse, sys, uuid, logging, logging.config, re, shutil
from time import strftime
import numpy as np, uproot
import awkward as ak
import seutils

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


_scripts = {}
def is_script(fn):
    _scripts[fn.__name__] = fn
    return fn


def uid():
    return str(uuid.uuid4())


def cleanup(path):
    if osp.isfile(path):
        logger.warning(f'Removing {path}')
        os.remove(path)


@contextmanager
def make_remote_file_local(remote, local=None):
    must_cleanup = False
    try:
        if not seutils.path.has_protocol(remote):
            # Filename is already local; do nothing
            yield remote, remote
        else:
            # Filename is remote; copy it to local, clean up local later
            if local is None: local = uid()
            logger.info('Copying %s -> %s', remote, local)
            seutils.cp(remote, local)
            must_cleanup = True
            yield local, remote
    finally:
        if must_cleanup: cleanup(local)
    


def trig_eff(tree, required_triggers):
    all_triggers = tree['TriggerPass'].title.split(',')
    required_decision_indices = ([ all_triggers.index(t) for t in required_triggers ])
    decisions = tree['TriggerPass'].array().to_numpy()
    decisions = decisions[:, required_decision_indices]
    assert decisions.shape == (tree.num_entries, len(required_triggers))
    passes = np.any(decisions, axis=1)
    assert passes.shape == (tree.num_entries,)
    return passes, decisions


def expand_wildcards(pats):
    """
    Expands * into full filenames
    """
    files = []
    for pat in pats:
        if '*' in pat:
            if seutils.path.has_protocol(pat):
                files.extend(seutils.ls_wildcard(pat))
            else:
                files.extend(glob.glob(pat))
        else:
            files.append(pat)
    return files


def load_treemaker_crosssection_txt():
    import requests
    cache = '/tmp/treemaker_xs.txt'
    if not osp.isfile('/tmp/treemaker_xs.txt'):
        url = 'https://raw.githubusercontent.com/TreeMaker/TreeMaker/Run2_UL/WeightProducer/python/MCSampleValues.py'
        text = requests.get(url).text
        with open(cache, 'w') as f:
            text = text.lower()
            f.write(text)
            return text
    else:
        with open(cache) as f:
            return f.read()


class Record(dict):
    @property
    def xs(self):
        return self['crosssection']['xs_13tev']

    @property
    def br(self):
        try:
            return self['branchingratio']['br_13tev']
        except KeyError:
            return 1.

    @property
    def kfactor(self):
        if 'kfactor' in self:
            for key, val in self['kfactor'].items():
                if key.startswith('kfactor_'):
                    return val
        return 1.

    @property
    def effxs(self):
        return self.xs*self.br*self.kfactor


def get_record(key):
    text = load_treemaker_crosssection_txt()
    match = re.search('"'+key+'"' + r' : ({[\w\W]*?})', text, re.MULTILINE)
    if not match: raise Exception(f'Could not find record for {key}')
    record_txt = match.group(1).replace('xsvalues', 'dict').replace('brvalues', 'dict').replace('kfactorvalues', 'dict')
    return Record(eval(record_txt))


class Hadd:
    @classmethod
    def from_files(cls, npzfiles):
        hadd = cls.from_file(npzfiles[0])
        for npzfile in npzfiles[1:]:
            hadd.concat(cls.from_file(npzfile))
        return hadd

    @classmethod
    def from_file(cls, npzfile):
        logger.info(f'Loading from {npzfile}')
        d = seutils.load_npz(npzfile, allow_pickle=True)
        inst = cls()
        inst.metadata = d['metadata'].item()
        inst.arrays = d['arrays'].item()
        inst.metadata['npz'] = npzfile
        return inst

    def __init__(self):
        self.metadata = {}
        self.arrays = {}

    @property
    def n(self):
        for v in self.arrays.values():
            return len(v)

    def __getattr__(self, key):
        try:
            return self.arrays[key]
        except KeyError:
            raise AttributeError

    def __getitem__(self, where):
        new = Hadd()
        new.metadata = copy.deepcopy(self.metadata)
        new.arrays = { k : v[where] for k, v in self.arrays.items() }
        return new

    def concat(self, *hadds, **kwargs):
        # Merge other hadds in
        for other in hadds:
            for key, array in other.arrays.items():
                if key in self.arrays:
                    self.arrays[key] = np.concatenate((self.arrays[key], array))
                else:
                    self.arrays[key] = array
        # Merge keywords in
        for key, array in kwargs.items():
            if key in self.arrays:
                self.arrays[key] = np.concatenate((self.arrays[key], array))
            else:
                self.arrays[key] = array

    def save(self, outfile):
        logger.info(f'Saving to {outfile}')
        np.savez(outfile, metadata=self.metadata, arrays=self.arrays)

    def set_metadata_from_filename(self, path):
        self.metadata.update(metadata_from_filename(path))

    def trigger_eff_for_selection(self, selector, trig_pass_branch='trig_pass_wmet'):
        sel = self[selector(self)]
        return sel.arrays[trig_pass_branch].sum() / sel.n if sel.n>0 else 1.

    @property
    def mz(self):
        return self.metadata["mz"]

    @property
    def mdark(self):
        return self.metadata["mdark"]

    @property
    def rinv(self):
        return self.metadata["rinv"]

    @property
    def is_bkg(self):
        return 'bkg' in self.metadata

    @property
    def bkg(self):
        return self.metadata['bkg']

    @property
    def doublecounting_eff(self):
        return self.metadata.get('doublecounting_eff', 1.)

    def apply_doublecounting_filter(self):
        """
        Avoid double counting of ttbar events. Returns new instance of Hadd
        """
        if self.is_bkg:
            if self.bkg=='ttjets':
                channel = self.metadata['channel']
                madht = self.arrays['madht']
                genmet = self.arrays['genmet']
                nleptons = self.arrays['nleptons']
                if channel == 'inclusive':
                    if 'ht' in self.metadata:
                        selection = madht>600.
                    else:
                        selection = (madht<600.) & (nleptons==0)
                elif channel in ['dilept', 'singleleptfromtbar', 'singleleptfromt']:
                    if 'genmet' in self.metadata:
                        selection = (madht < 600.) & (genmet >= 150.)
                    else:
                        selection = (madht < 600.) & (genmet < 150.)
                new = self[selection]
                new.metadata['doublecounting_eff'] = selection.sum() / len(selection)
                return new
            elif self.bkg=='wjets' and not 'ht' in self.metadata:
                selection = self.arrays['madht']<100.
                new = self[selection]
                new.metadata['doublecounting_eff'] = selection.sum() / len(selection)
                return new
        return self

    @property
    def xs_record_key(self):
        if not self.is_bkg: raise Exception('Only for bkg!')
        meta = self.metadata
        key = meta['bkg']
        if meta['bkg'] == 'zjets':
            key += 'tonunu'
        if meta['bkg'] == 'wjets':
            key += 'tolnu'
        if 'channel' in meta and 'lept' in meta['channel']:
            key += '_' + meta['channel']
        if 'genmet' in meta:
            key += '_genmet-{}'.format(meta['genmet'])
        if 'ht' in meta:
            key += '_ht-{}to{}'.format(*meta['ht'])
        if 'pt' in meta:
            key += '_pt_{}to{}'.format(*meta['pt'])
        return key

    @property
    def xs_record(self):
        if not hasattr(self, '_xs_record'):
            self._xs_record = get_record(self.xs_record_key)
        return self._xs_record

    @property
    def xs(self):
        return self.xs_record.xs * self.xs_record.kfactor * self.xs_record.br * self.doublecounting_eff

    @property
    def label(self):
        if self.is_bkg:
            return self.xs_record_key
        else:
            return (
                f'mz{self.metadata["mz"]}_mdark{self.metadata["mdark"]}'
                f'_rinv{self.metadata["rinv"]:.1f}'
                )

    def trig_eff(self, variable, axis, trig_pass_branch='trig_pass_wmet', acc=True):
        axis = np.concatenate((axis, [np.inf])) # Final '>' should contain everything that's left
        x = self.arrays[variable]
        x_pass_trig = self.arrays[variable][self.arrays[trig_pass_branch]]
        n = x.shape[0]

        h_sel = np.histogram(x, axis)[0]
        sel = np.cumsum(h_sel[::-1])[::-1]
        hsel_and_trig = np.histogram(x_pass_trig, axis)[0]
        sel_and_trig = np.cumsum(hsel_and_trig[::-1])[::-1]
        eff = np.where(sel!=0., sel_and_trig/sel, 0.)
        # do not allow decreases, always use maximum up until that point
        if acc: eff = np.maximum.accumulate(eff)
        return eff


class MultipleHadds:
    def __init__(self, hadds):
        self.hadds = hadds

    @property
    def xs(self):
        return sum(h.xs for h in self.hadds)

    def trig_eff(self, variable, axis, trig_pass_branch='trig_pass_wmet'):
        eff = np.zeros_like(axis)
        for hadd in self.hadds:
            eff += hadd.xs/self.xs * hadd.trig_eff(variable, axis, trig_pass_branch)
        return eff

    def trigger_eff_for_selection(self, selector, trig_pass_branch='trig_pass_wmet'):
        """
        Simply a weighted version of trigger_eff_for_selection for single Hadd instances
        """
        pass_selection = 0.
        pass_selection_and_trigger = 0.
        eff = 0.
        for hadd in self.hadds:
            sel = hadd[selector(hadd)]
            n_sel = sel.n
            n_sel_and_trig = sel.arrays[trig_pass_branch].sum()
            weight = hadd.xs / self.xs
            pass_selection += weight * sel.n
            pass_selection_and_trigger += weight * sel.arrays[trig_pass_branch].sum()
            eff += weight * (n_sel_and_trig / n_sel if n_sel>0 else 1.)

        return pass_selection_and_trigger / pass_selection if pass_selection>0. else 0.


def metadata_from_filename(path):
    """
    Extracts physics parameters from a path and stores in a metadata dict.
    """
    metadata = {}
    # Signal metadata
    match = re.search(r'mz(\d+)', path)
    if match:
        metadata['mz'] = int(match.group(1))
        logger.info(f'Setting mz={metadata["mz"]}')        
    match = re.search(r'mdark(\d+)', path)
    if match:
        metadata['mdark'] = int(match.group(1))
        logger.info(f'Setting mdark={metadata["mdark"]}')
    match = re.search(r'rinv(\d\.\d+)', path)
    if match:
        metadata['rinv'] = float(match.group(1))
        logger.info(f'Setting rinv={metadata["rinv"]}')
    # Bkg metadata
    for bkg in ['qcd', 'ttjets', 'zjets', 'wjets']:
        if bkg in ('/'.join(path.rsplit('/',3)[1:])).lower():
            metadata['bkg'] = bkg
            logger.info('Determined bkg=%s', bkg)
            if bkg == 'ttjets':
                for ch in ['SingleLeptFromTbar', 'SingleLeptFromT', 'DiLept']:
                    if ch in path:
                        metadata['channel'] = ch.lower()
                        break
                else:
                    metadata['channel'] = 'inclusive'
                logger.info('Determined channel=%s', metadata['channel'])
            break
    match = re.search(r'HT\-(\d+)[tT]o([\dInf]+)', path)
    if match:
        left = int(match.group(1))
        right = np.inf if match.group(2) == 'Inf' else int(match.group(2))
        metadata['ht'] = (left, right)
        logger.info(f'Setting ht=({metadata["ht"]})')
    match = re.search(r'Pt_(\d+)to([\dInf]+)', path)
    if match:
        left = int(match.group(1))
        right = np.inf if match.group(2) == 'Inf' else int(match.group(2))
        metadata['pt'] = (left, right)
        logger.info(f'Setting pt=({metadata["pt"]})')
    match = re.search(r'genMET\-(\d+)', path)
    if match:
        metadata['genmet'] = int(match.group(1))
        logger.info('Setting genmet=%s', metadata['genmet'])
    match = re.search(r'madpt(\d+)', path)
    if match:
        metadata['madpt'] = int(match.group(1))
        logger.info('Setting madpt=%s', metadata['madpt'])
    match = re.search(r'genjetpt(\d+)', path)
    if match:
        metadata['genjetpt'] = int(match.group(1))
        logger.info('Setting genjetpt=%s', metadata['genjetpt'])
    return metadata


def filename_from_metadata(metadata, ext='.npz'):
    """
    Generate a filename for an Hadd class instance.
    """
    if 'bkg' in metadata:
        f = metadata['bkg']
        if 'channel' in metadata:
            f += '_' + metadata['channel']
        if 'genmet' in metadata:
            f += f'_genmet-{metadata["genmet"]}'
        if 'pt' in metadata:
            f += '_pt{}-{}'.format(*metadata['pt'])
        if 'ht' in metadata:
            f += '_ht{}-{}'.format(*metadata['ht'])
    else:
        f = (
            f'mz{metadata["mz"]}_mdark{metadata["mdark"]}'
            f'_rinv{metadata["rinv"]:.1f}'
            # f'_{"wmet" if metadata["wmet"] else "nomet"}'
            )
        if 'madpt' in metadata:
            f = f'madpt{metadata["madpt"]}_' + f
        if 'genjetpt' in metadata:
            f = f'genjetpt{metadata["genjetpt"]}_' + f
    f += ext
    logger.info('Compiled filename %s', f)
    return f


def hadd_factory(rootfiles, nmax=None):
    out = Hadd()
    ntodo = nmax
    for rootfile in rootfiles:
        with make_remote_file_local(rootfile) as (local, remote):
            tree = uproot.open(local)['TreeMaker2/PreSelection']
            arrays = tree.arrays([
                'JetsAK8.fCoordinates.fPt', 'HT', 'MET', 'GenMET', 'GenParticles_PdgId', 'madHT'
                ])
            jetpt_akarray = arrays['JetsAK8.fCoordinates.fPt']
            njets = ak.to_numpy(jetpt_akarray.layout.count())
            n_events = njets.shape[0]
            # jetpt: fill in -1 whereever there were 0 jets
            jetpt = -1.*np.ones(njets.shape)
            jetpt[njets>0] = jetpt_akarray[njets>0,0].to_numpy()
            trig_pass_wmet, _ = trig_eff(tree, triggers_2018_wmet)
            trig_pass_nomet, _ = trig_eff(tree, triggers_2018_nomet)

            # Count leptons per event
            n_leptons = np.zeros(njets.shape[0])
            pid = np.abs(arrays['GenParticles_PdgId'])
            for id in [11, 13, 15]:
                n_leptons += ak.sum(pid==id, axis=-1).to_numpy()
            assert n_leptons.shape == (n_events,)

            out.concat(
                njets=njets, jetpt=jetpt,
                ht=arrays['HT'].to_numpy(), met=arrays['MET'].to_numpy(),
                genmet=arrays['GenMET'].to_numpy(), madht=arrays['madHT'].to_numpy(),
                nleptons=n_leptons,
                trig_pass_wmet=trig_pass_wmet, trig_pass_nomet=trig_pass_nomet
                )
            if nmax is not None:
                ntodo -= njets.shape[0]
                if ntodo <= 0:
                    logger.info('Reached %s required events, stop loop', nmax)
                    break

    out.metadata['rootfiles'] = rootfiles
    out.set_metadata_from_filename(rootfiles[0])
    return out


triggers_2018_nomet = [
    # AK8PFJet triggers
    'HLT_AK8PFJet500_v',
    'HLT_AK8PFJet550_v',
    # CaloJet
    'HLT_CaloJet500_NoJetID_v',
    'HLT_CaloJet550_NoJetID_v',
    # PFJet and PFHT
    'HLT_PFHT1050_v', # but, interestingly, not HLT_PFHT8**_v or HLT_PFHT9**_v, according to the .txt files at least
    'HLT_PFJet500_v',
    'HLT_PFJet550_v',
    # Trim mass jetpt+HT
    'HLT_AK8PFHT800_TrimMass50_v',
    'HLT_AK8PFHT850_TrimMass50_v',
    'HLT_AK8PFHT900_TrimMass50_v',
    'HLT_AK8PFJet400_TrimMass30_v',
    'HLT_AK8PFJet420_TrimMass30_v',
    ]

triggers_2018_wmet = triggers_2018_nomet + [
    # MET triggers
    'HLT_PFHT500_PFMET100_PFMHT100_IDTight_v',
    'HLT_PFHT500_PFMET110_PFMHT110_IDTight_v',
    'HLT_PFHT700_PFMET85_PFMHT85_IDTight_v',
    'HLT_PFHT700_PFMET95_PFMHT95_IDTight_v',
    'HLT_PFHT800_PFMET75_PFMHT75_IDTight_v',
    'HLT_PFHT800_PFMET85_PFMHT85_IDTight_v',
    ]


def worker(input):
    rootfiles, outdir, nmax = input
    hadd = hadd_factory(rootfiles, nmax=nmax)
    dst = osp.join(outdir, filename_from_metadata(hadd.metadata))
    if not osp.isdir(outdir): os.makedirs(outdir)
    hadd.save(dst)

@is_script
def make_npzs_new():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rootfiles', nargs='+', action='append')
    parser.add_argument('-o', '--outdir', type=str, default='data')
    parser.add_argument('-n', '--nmax', type=int, default=None)
    args = parser.parse_args()

    print(args.rootfiles)

    mp_args = []
    for subset in args.rootfiles:
        subset = expand_wildcards(subset)
        mp_args.append([subset, args.outdir, args.nmax])

    print(mp_args)

    import multiprocessing as mp
    p = mp.Pool(8)
    p.map(worker, mp_args)
    p.close()
    p.join()


@is_script
def make_npzs_bkg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outdir', type=str, default='data_bkg')
    parser.add_argument('-n', '--nmax', type=int, default=None)
    args = parser.parse_args()

    base = 'root://cmseos.fnal.gov//store/user/lpcsusyhad/SusyRA2Analysis2015/Run2ProductionV20/Summer20UL18/'
    bkg_dirs = []
    bkg_dirs.extend(seutils.ls_wildcard(base + 'QCD_Pt_170*'))
    # bkg_dirs.extend(seutils.ls_wildcard(base + 'TTJets*'))
    # bkg_dirs.extend(seutils.ls_wildcard(base + 'ZJets*'))
    # bkg_dirs.extend(seutils.ls_wildcard(base + 'WJetsToLNu*'))

    mp_args = [ [seutils.ls_wildcard(d+'/*.root'), args.outdir, args.nmax] for d in bkg_dirs ]
    logger.info(f'Processing {sum(len(t[0]) for t in mp_args)} rootfiles from {len(bkg_dirs)} directories')

    import multiprocessing as mp
    p = mp.Pool(8)
    p.map(worker, mp_args)
    p.close()
    p.join()


def make_npz_worker(input):
    rootfile = input[0]
    # if isinstance(rootfile, str): rootfile = [rootfile]
    outdir = input[1]
    src_rootfile = rootfile
    logger.info(f'Working on {rootfile}')
    with make_remote_file_local(rootfile) as (local, remote):        
        if not osp.isdir(outdir): os.makedirs(outdir)
        for wmet in [True, False]:
            triggers = triggers_2018_wmet if wmet else triggers_2018_nomet
            hadd = hadd_factory([local], triggers)
            hadd.set_metadata_from_filename(remote)
            hadd.metadata['wmet'] = wmet
            dst = osp.join(
                outdir,
                f'mz{hadd.metadata["mz"]}_mdark{hadd.metadata["mdark"]}'
                f'_rinv{hadd.metadata["rinv"]:.1f}_{"wmet" if wmet else "nomet"}.npz'
                )
            hadd.save(dst)



@is_script
def make_npzs():
    parser = argparse.ArgumentParser()
    parser.add_argument('rootfiles', nargs='+', type=str)
    parser.add_argument('-o', '--outdir', type=str, default='data')
    parser.add_argument('-m', '--multiprocessing', action='store_true')
    args = parser.parse_args()
    
    rootfiles = expand_wildcards(args.rootfiles)
    if not args.multiprocessing:
        for rootfile in rootfiles:
            make_npz_worker((rootfile, args.outdir))
    else:
        import multiprocessing as mp
        p = mp.Pool(10)
        p.map(make_npz_worker, [[rootfile, args.outdir] for rootfile in rootfiles])
        p.close()
        p.join()


@is_script
def trigger_plots():
    parser = argparse.ArgumentParser()
    parser.add_argument('npzs', nargs='+', type=str)
    parser.add_argument('-o', '--outdir', type=str, default=strftime('plots_%b%d'))
    args = parser.parse_args()

    hadds = [ Hadd.from_file(npz) for npz in args.npzs ]
    hadds.sort(key=lambda h: h.mz)
    wmet = hadds[0].metadata['wmet']
    wmet_label = '_wmet' if wmet else '_nomet'

    # ________________________________________________________
    # ptjet

    fig = plt.figure(figsize=(10,10))
    ax = fig.gca()
    jetpt_axis = np.linspace(0., 700., 100)

    for hadd in hadds:
        eff_jetpt = [
            hadd.trigger_eff_for_selection((hadd.njets>0) & (hadd.jetpt > jetpt))
            for jetpt in jetpt_axis
            ]
        dots = ax.plot(
            jetpt_axis, eff_jetpt, '.',
            label=f'$m_{{Z\\prime}}$ = {hadd.mz}; $r_{{inv}}$ = {hadd.rinv:.1f}',
            )[0]

    ax.legend()
    ax.set_xlabel(r'$p_{T}^{jet}$ (GeV)')
    ax.set_ylabel(r'$N_{pass}/N_{total}$')

    if not osp.isdir(args.outdir): os.makedirs(args.outdir)
    outfile = osp.join(args.outdir, 'jetpt' + wmet_label + '.png')
    plt.savefig(outfile, bbox_inches='tight')
    os.system('imgcat ' + outfile)

    # ________________________________________________________
    # HT

    fig = plt.figure(figsize=(10,10))
    ax = fig.gca()
    ht_axis = np.linspace(0., 1200., 100)

    for hadd in hadds:
        eff_ht = [hadd.trigger_eff_for_selection(hadd.ht>ht) for ht in ht_axis]
        ax.plot(
            ht_axis, eff_ht, '.',
            label=f'$m_{{Z\\prime}}$ = {hadd.mz}; $r_{{inv}}$ = {hadd.rinv:.1f}',
            )

    ax.legend()
    ax.set_xlabel(r'$H_{T}$ (GeV)')
    ax.set_ylabel(r'$N_{pass}/N_{total}$')
    outfile = osp.join(args.outdir, 'ht' + wmet_label + '.png')
    plt.savefig(outfile, bbox_inches='tight')
    os.system('imgcat ' + outfile)

    # ________________________________________________________
    # MET

    fig = plt.figure(figsize=(10,10))
    ax = fig.gca()
    met_axis = np.linspace(0., 1200., 100)

    for hadd in hadds:
        eff_met = [hadd.trigger_eff_for_selection(hadd.met>met)for met in met_axis]
        ax.plot(
            met_axis, eff_met, '.',
            label=f'$m_{{Z\\prime}}$ = {hadd.mz}; $r_{{inv}}$ = {hadd.rinv:.1f}',
            )

    ax.legend()
    ax.set_xlabel(r'MET (GeV)')
    ax.set_ylabel(r'$N_{pass}/N_{total}$')
    outfile = osp.join(args.outdir, 'met' + wmet_label + '.png')
    plt.savefig(outfile, bbox_inches='tight')
    os.system('imgcat ' + outfile)


@is_script
def bkg_eff():
    parser = argparse.ArgumentParser()
    parser.add_argument('npzs', nargs='+', type=str)
    parser.add_argument('-o', '--outdir', type=str, default=strftime('plots_%b%d'))
    args = parser.parse_args()

    bkgs = ['qcd', 'ttjets', 'wjets', 'zjets']
    all_hadds = [ Hadd.from_file(npz) for npz in args.npzs ]
    # Filter out the low pt QCD samples
    all_hadds = [ h for h in all_hadds if h.metadata.get('pt', [1e9])[0]>150. ]

    # Group them by bkg type (qcd/ttjets/wjets/zjets)
    hadds = []
    for bkg in bkgs:
        hadds_this_bkg = [h for h in all_hadds if h.bkg==bkg]
        if len(hadds_this_bkg) > 0:
            hadds.append(MultipleHadds(hadds_this_bkg))

    # ________________________________________________________
    # ptjet

    fig = plt.figure(figsize=(10,10))
    ax = fig.gca()
    jetpt_axis = np.linspace(0., 700., 100)

    for hadd in hadds:
        eff_jetpt = hadd.trig_eff('jetpt', jetpt_axis)
        line = ax.plot(
            jetpt_axis, eff_jetpt, '.',
            label=hadd.hadds[0].bkg,
            )[0]
        try:
            interpolation = Interpolation(jetpt_axis, eff_jetpt)
            ax.plot(*interpolation.fine(), c=line.get_color())
            line.set_label(line.get_label() + f' 98%={interpolation.solve(.98):.1f} GeV')
        except ValueError:
            pass

    ax.legend()
    ax.set_xlabel(r'$p_{T}^{jet}$ (GeV)')
    ax.set_ylabel(r'$N_{pass}/N_{total}$')

    if not osp.isdir(args.outdir): os.makedirs(args.outdir)
    outfile = osp.join(args.outdir, 'jetpt_bkg.png')
    plt.savefig(outfile, bbox_inches='tight')
    os.system('imgcat ' + outfile)


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



def dump_eff_worker(input):
    bkg_dir, outdir, cut_varname, axis = input
    logger.info(f'Starting work on {bkg_dir}')
    hadd = Hadd.from_files(seutils.ls_wildcard(bkg_dir + '/*.npz'))
    eff_jetpt_wmet_noacc = hadd.trig_eff(cut_varname, axis, 'trig_pass_wmet', acc=False)
    eff_jetpt_nomet_noacc = hadd.trig_eff(cut_varname, axis, 'trig_pass_nomet', acc=False)
    eff_jetpt_wmet = np.maximum.accumulate(eff_jetpt_wmet_noacc)
    eff_jetpt_nomet = np.maximum.accumulate(eff_jetpt_nomet_noacc)
    dst = osp.join(outdir, filename_from_metadata(hadd.metadata, ext='.eff'))
    logger.info(f'Dumping to {dst}')
    np.savez(
        dst,
        eff_wmet_noacc = eff_jetpt_wmet_noacc,
        eff_nomet_noacc = eff_jetpt_nomet_noacc,
        eff_wmet = eff_jetpt_wmet,
        eff_nomet = eff_jetpt_nomet,
        xs=hadd.xs, axis=axis
        )

@is_script
def dump_jetpt_bkg_effs():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    bkg_dirs = seutils.ls_wildcard('root://cmseos.fnal.gov//store/user/lpcdarkqcd/triggerstudy/bkg/FLATC/*')

    jetpt_axis = np.linspace(0., 700., 100)

    outdir = 'bkg_eff/'
    if not osp.isdir(outdir): os.makedirs(outdir)

    mp_args = [
        [bkg_dir, outdir, 'jetpt', jetpt_axis]
        for bkg_dir in bkg_dirs
        ]

    import multiprocessing as mp
    p = mp.Pool(8)
    p.map(dump_eff_worker, mp_args)
    p.close()
    p.join()


@is_script
def plot_jetpt_fromeffs():
    parser = argparse.ArgumentParser()
    parser.add_argument('npzs', nargs='+', type=str)
    parser.add_argument('-o', '--outfile', type=str, default='fromeffs.png')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-g', '--group', action='store_true')
    parser.add_argument('-f', '--fit', action='store_true')
    parser.add_argument('--nomet', action='store_true')
    parser.add_argument('--noacc', action='store_true')
    args = parser.parse_args()

    varname = 'eff_' + ('nomet' if args.nomet else 'wmet') + ('_noacc' if args.noacc else '')

    effs = []
    xss = []
    for npz in args.npzs:
        d = np.load(npz)
        eff = d[varname]
        if np.max(eff) < .98: logger.warning(f'{npz} efficiency only reaches {np.max(eff)}')
        effs.append(eff)
        xss.append(d['xs'])

    # Combine the effs
    effs = np.vstack(effs)
    xss = np.array(xss)
    weights = xss / xss.sum()
    eff = (effs * np.expand_dims(weights,-1)).sum(axis=0)

    fig = plt.figure(figsize=(10,10))
    ax = fig.gca()
    jetpt_axis = np.linspace(0., 700., 100)
    line = ax.plot(jetpt_axis, eff, '.')[0]
    if args.fit:
        try:
            interpolation = Interpolation(jetpt_axis, eff)
            interp_line = ax.plot(*interpolation.fine(), c=line.get_color())[0]
            interp_line.set_label(f'98%={interpolation.solve(.98):.1f} GeV')
        except ValueError:
            logger.error(f'Could not interpolate')

    ax.legend()
    ax.set_xlabel(r'$p_{T}^{jet}$ (GeV)')
    ax.set_ylabel(r'$N_{pass}/N_{total}$')

    outdir = osp.dirname(osp.abspath(args.outfile))
    if not osp.isdir(outdir): os.makedirs(outdir)
    plt.savefig(args.outfile, bbox_inches='tight')
    os.system('imgcat ' + args.outfile)




@is_script
def plot_jetptdist():
    parser = argparse.ArgumentParser()
    parser.add_argument('npzs', nargs='+', type=str)
    parser.add_argument('-o', '--outfile', type=str, default='jetptdist.png')
    parser.add_argument('--highptzoomin', action='store_true')
    args = parser.parse_args()

    fig = plt.figure(figsize=(10,10))
    ax = fig.gca()
    jetpt_axis = np.linspace(0., 600., 40)
    if args.highptzoomin: jetpt_axis = np.linspace(500., 800., 40)

    hadds = [ Hadd.from_file(npz) for npz in expand_wildcards(args.npzs) ]
    hadds.sort(key=lambda h: h.metadata['pt'][0] if 'pt' in h.metadata else 0)

    for hadd in hadds:
        ax.hist(
            hadd.arrays['jetpt'], jetpt_axis,
            label=osp.basename(hadd.metadata['npz']).replace('.npz',''),
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



@is_script
def plot_jetpt():
    parser = argparse.ArgumentParser()
    parser.add_argument('npzs', nargs='+', type=str)
    parser.add_argument('-o', '--outfile', type=str, default='jetpt.png')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-g', '--group', action='store_true')
    parser.add_argument('-f', '--fit', action='store_true')
    parser.add_argument('-b', '--trigpass', type=str, default='trig_pass_wmet')
    args = parser.parse_args()

    fig = plt.figure(figsize=(10,10))
    ax = fig.gca()
    jetpt_axis = np.linspace(0., 700., 100)

    hadds = [ Hadd.from_file(npz) for npz in expand_wildcards(args.npzs) ]
    hadds.sort(key=lambda h: h.metadata['pt'][0] if 'pt' in h.metadata else 0)

    if args.group:
        hadds = [MultipleHadds(hadds)]
        hadds[0].label = 'group'

    for hadd in hadds:
        eff_jetpt = hadd.trig_eff('jetpt', jetpt_axis, args.trigpass)
        line = ax.plot(
            jetpt_axis, eff_jetpt, '.',
            label = hadd.metadata['npz'].split('/')[1].replace('.npz','')
            # label=hadd.label
            )[0]

        if args.fit:
            try:
                interpolation = Interpolation(jetpt_axis, eff_jetpt)
                ax.plot(*interpolation.fine(), c=line.get_color())
                line.set_label(line.get_label() + f' 98%={interpolation.solve(.98):.1f} GeV')
            except ValueError:
                logger.error(f'Could not interpolate {hadd.label}')

        if args.debug:
            print(hadd.label)
            for jetpt in jetpt_axis:
                sel = hadd[(hadd.njets>0) & (hadd.jetpt > jetpt)]
                n_sel_and_trig = sel.arrays[args.trigpass].sum()
                n_sel = sel.arrays[args.trigpass].shape[0]
                print(
                    f'  jetpt={jetpt:>4.0f} GeV  sel&trig={n_sel_and_trig:7}  sel={n_sel:7}'
                    f'  eff={n_sel_and_trig/n_sel if n_sel>0 else 0.:.3f}'
                    )

    ax.legend(fontsize=18)
    ax.set_xlabel(r'$p_{T}^{jet}$ (GeV)')
    ax.set_ylabel(r'$N_{pass}/N_{total}$')

    outdir = osp.dirname(osp.abspath(args.outfile))
    if not osp.isdir(outdir): os.makedirs(outdir)
    plt.savefig(args.outfile, bbox_inches='tight')
    os.system('imgcat ' + args.outfile)


@is_script
def plot_jetpt_pizza():
    parser = argparse.ArgumentParser()
    parser.add_argument('npzs', nargs='+', type=str)
    parser.add_argument('-o', '--outfile', type=str, default='jetpt.png')
    parser.add_argument('-b', '--trigpass', type=str, default='trig_pass_wmet')
    args = parser.parse_args()

    set_mpl_fontsize(30,34,38)
    fig = plt.figure(figsize=(10,10))
    ax = fig.gca()
    jetpt_axis = np.linspace(0., 700., 100)

    hadds = [ Hadd.from_file(npz) for npz in expand_wildcards(args.npzs) ]
    hadds.sort(key=lambda h: h.metadata['pt'][0] if 'pt' in h.metadata else 0)

    def label(hadd):
        m = hadd.metadata
        return rf'$m_{{Z\prime}}={m["mz"]:.0f}$, $r_{{inv}}={m["rinv"]:.1f}$'

    for hadd in hadds:
        eff_jetpt = hadd.trig_eff('jetpt', jetpt_axis, args.trigpass)
        line = ax.plot(
            jetpt_axis, eff_jetpt, '.',
            label = label(hadd)
            )[0]

        x_98_vals = []
        try:
            interpolation = Interpolation(jetpt_axis, eff_jetpt)
            ax.plot(*interpolation.fine(), c=line.get_color())
            x_98_vals.append(interpolation.solve(.98))
            # line.set_label(line.get_label() + f' 98%={interpolation.solve(.98):.1f} GeV')
        except ValueError:
            logger.error(f'Could not interpolate {hadd.label}')

    mean_98 = sum(x_98_vals)/len(x_98_vals)
    ax.set_ylim(-.05, 1.05)
    ax.plot([mean_98, mean_98], ax.get_ylim(), '--', c='black')

    leg = ax.legend(fontsize=20, loc="upper left", framealpha=.0)

    for h in leg.legendHandles: h._legmarker.set_markersize(20)

    ax.set_xlabel(r'$p_{T}^{jet}$ (GeV)')
    ax.set_ylabel(r'$N_{pass}/N_{total}$')

    outdir = osp.dirname(osp.abspath(args.outfile))
    if not osp.isdir(outdir): os.makedirs(outdir)
    plt.savefig(args.outfile, bbox_inches='tight')
    os.system('imgcat ' + args.outfile)

def se_download_worker(input):
    npz_files, outdir = input
    logger.info(f'Working on {len(npz_files)} npz files, first one being {npz_files[0]}, outdir={outdir}')
    hadd = Hadd.from_file(npz_files[0])
    for npz_file in npz_files[1:]:
        hadd.concat(Hadd.from_file(npz_file))
    dst = osp.join(outdir, filename_from_metadata(hadd.metadata))
    hadd.save(dst)

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

@is_script
def download_full_bkg():
    bkg_dirs = seutils.ls_wildcard('root://cmseos.fnal.gov//store/user/lpcdarkqcd/triggerstudy/bkg/FLATC/*')
    outdir = 'data_bkg_fromse'
    if not osp.isdir(outdir): os.makedirs(outdir)
    mp_args = [[seutils.ls_wildcard(bkg_dir+'/*.npz'), outdir] for bkg_dir in bkg_dirs]

    print(mp_args[0])

    import multiprocessing as mp
    p = mp.Pool(8)
    p.map(se_download_worker, mp_args)
    p.close()
    p.join()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('script', type=str, choices=list(_scripts.keys()))
    parser.add_argument('-v', '--verbose', action='store_true')
    global_args, other_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + other_args
    r = _scripts[global_args.script]()