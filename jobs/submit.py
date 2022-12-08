import os.path as osp
from time import strftime
import seutils
from jdlfactory import Group
group = Group.from_file('worker.py')
group.venv(py3=True)
group.sh('pip install uproot awkward seutils')
group.sh('pip install https://github.com/boostedsvj/svj_ntuple_processing/archive/main.zip')

group.group_data['stageout'] = strftime('root://cmseos.fnal.gov//store/user/lpcdarkqcd/triggerstudy/bkg_%b%d/TRIGCOL/')

base = 'root://cmseos.fnal.gov//store/user/lpcsusyhad/SusyRA2Analysis2015/Run2ProductionV20/Summer20UL18/'
bkg_dirs = []
bkg_dirs.extend(seutils.ls_wildcard(base + 'QCD_Pt_*'))
bkg_dirs.extend(seutils.ls_wildcard(base + 'TTJets*'))
bkg_dirs.extend(seutils.ls_wildcard(base + 'ZJets*'))
bkg_dirs.extend(seutils.ls_wildcard(base + 'WJetsToLNu*'))

def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

for bkg_dir in bkg_dirs:
    for chunk in divide_chunks(seutils.ls_wildcard(bkg_dir+'/*.root'), 600):
        group.add_job(dict(rootfiles=chunk))

# group.run_locally()
group.prepare_for_jobs('bkgcols')