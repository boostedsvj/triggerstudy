import os.path as osp, argparse
from time import strftime
import seutils
from jdlfactory import Group, logger

group = Group.from_file('worker.py')
group.venv(py3=True)
group.sh('pip install uproot awkward seutils')
group.sh('pip install https://github.com/boostedsvj/svj_ntuple_processing/archive/main.zip')

group.group_data['stageout'] = 'root://cmseos.fnal.gov//store/user/lpcdarkqcd/triggerstudy/bkg_Dec08/TRIGCOL/'

base = 'root://cmseos.fnal.gov//store/user/lpcsusyhad/SusyRA2Analysis2015/Run2ProductionV20/Summer20UL18/'
cache_file = 'cache_rootfiles.txt'

def get_rootfiles():
    if not osp.isfile(cache_file):
        logger.info('Retrieving list of rootfiles...')
        rootfiles = []
        for pat in ['QCD_Pt_*', 'TTJets*', 'ZJets*', 'WJetsToLNu*']:
            rootfiles.extend(seutils.ls_wildcard(base + pat + '/*.root'))
        with open(cache_file, 'w') as f:
            f.write('\n'.join(rootfiles))
        logger.info('Wrote cached list to %s', cache_file)
        return rootfiles
    else:
        logger.info('Using cached list of rootfiles from %s', cache_file)
        with open(cache_file) as f:
            rootfiles = [l.strip() for l in f.readlines()]
        return rootfiles

def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--listmissing', action='store_true')
    args = parser.parse_args()

    rootfiles = get_rootfiles()

    if args.listmissing:
        dsts = [
            osp.join(
                group.group_data['stageout'], '/'.join(r.rsplit('/',3)[1:])
                ).replace('.root','.npz')
            for r in rootfiles
            ]
        logger.info('Missing output files:')
        for dst in dsts:
            if not seutils.isfile(dst): logger.warning('File %s does not exist', dst)
        logger.info('Done listing')
        return 

    for chunk in divide_chunks(rootfiles, 600):
        group.add_job(dict(rootfiles=chunk))

    # group.run_locally()
    group.prepare_for_jobs('bkgcols')

if __name__ == '__main__':
    main()