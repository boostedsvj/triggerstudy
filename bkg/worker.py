import os, os.path as osp, re
import seutils, numpy as np
import svj_ntuple_processing as svj
from jdlfactory_server import data, group_data, ijob # type: ignore

for rootfile in data.rootfiles:
    dst = osp.join(group_data.stageout, '/'.join(rootfile.rsplit('/',3)[1:])).replace('.root','.npz')
    if seutils.isfile(dst):
        svj.logger.info(f'File {dst} exists, skipping')
        continue
    array = svj.open_root(rootfile)
    array = svj.filter_stitch(array)
    array = svj.filter_at_least_one_ak8jet(array)
    cols = svj.triggerstudy_columns(array)
    cols.metadata.update(svj.metadata_from_filename('/'.join(rootfile.rsplit('/',3)[1:])))
    cols.save(dst)
