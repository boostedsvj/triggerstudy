import re, os.path as osp, argparse, glob

import seutils
import svj_ntuple_processing as svj


def expand_wildcards(pats):
    expanded = []
    for pat in pats:
        if '*' in pat:
            if seutils.path.has_protocol(pat):
                expanded.extend(seutils.ls_wildcard(pat))
            else:
                expanded.extend(glob.glob(pat))
        else:
            expanded.append(pat)
    return expanded


def process_rootfile(tup):
    rootfile, dst = tup
    array = svj.open_root(rootfile)
    array = svj.filter_stitch(array)
    array = svj.filter_at_least_one_ak8jet(array)
    cols = svj.triggerstudy_columns(array)
    cols.metadata.update(svj.metadata_from_filename(rootfile))
    cols.save(dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dst', type=str, default='.')
    parser.add_argument('rootfiles', nargs='+', type=str)
    parser.add_argument('-n', '--nthreads', default=10, type=int)
    args = parser.parse_args()

    rootfiles = expand_wildcards(args.rootfiles)

    fn_args = []
    for rootfile in rootfiles:
        dst = osp.join(args.dst, osp.basename(rootfile).replace('.root', '.npz'))
        if seutils.path.has_protocol(dst) and seutils.isfile(dst):
            svj.logger.info('File %s exists, skipping', dst)
            continue
        fn_args.append((rootfile, dst))

    if len(fn_args) == 1:
        process_rootfile(fn_args[0])
    else:
        import multiprocessing as mp
        p = mp.Pool(args.nthreads)
        p.map(process_rootfile, fn_args)
        p.close()
        p.join()


if __name__ == '__main__':
    main()