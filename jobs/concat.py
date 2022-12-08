import os, os.path as osp, argparse
import seutils
import svj_ntuple_processing as svj

def worker(directory):
    directory = directory.rstrip('/')
    dst = directory.replace('TRIGCOL', 'TRIGCOLHADD') + '.npz'
    if seutils.isfile(dst):
        svj.logger.info(f'File {dst} exists, skipping')
        return
    rootfiles = seutils.ls_wildcard(directory + '/*.npz')
    combined = svj.concat_columns([svj.Columns.load(f) for f in rootfiles])
    combined.save(dst)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directories', type=str, nargs='+')
    parser.add_argument('--nthreads', type=int, default=8)
    args = parser.parse_args()

    directories = seutils.expand_wildcards(args.directories)

    if len(directories) == 1:
        worker(directories[0])
    else:
        import multiprocessing as mp
        pool = mp.Pool(args.nthreads)
        pool.map(worker, directories)
        pool.close()
        pool.join()


if __name__ == '__main__':
    main()