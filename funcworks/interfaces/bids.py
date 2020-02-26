import os
import shutil
from gzip import GzipFile
from nipype.utils.filemanip import copyfile
from nipype.interfaces.base import (
    BaseInterfaceInputSpec, TraitedSpec, SimpleInterface,
    InputMultiPath, OutputMultiPath, File, Directory,
    traits, isdefined
    )
from nipype.interfaces.io import IOBase
import nibabel as nb
from ..utils import snake_to_camel

def bids_split_filename(fname):
    """Split a filename into parts: path, base filename, and extension
    Respects multi-part file types used in BIDS standard and draft extensions
    Largely copied from nipype.utils.filemanip.split_filename
    Parameters
    ----------
    fname : str
        file or path name
    Returns
    -------
    pth : str
        path of fname
    fname : str
        basename of filename, without extension
    ext : str
        file extension of fname
    """
    special_extensions = [
        ".R.surf.gii", ".L.surf.gii",
        ".R.func.gii", ".L.func.gii",
        ".nii.gz", ".tsv.gz",
        ]

    pth = os.path.dirname(fname)
    fname = os.path.basename(fname)

    for special_ext in special_extensions:
        if fname.lower().endswith(special_ext.lower()):
            ext_len = len(special_ext)
            ext = fname[-ext_len:]
            fname = fname[:-ext_len]
            break
    else:
        fname, ext = os.path.splitext(fname)

    return pth, fname, ext

class BIDSDataSinkInputSpec(BaseInterfaceInputSpec):
    base_directory = Directory(
        mandatory=True,
        desc='Path to BIDS (or derivatives) root directory')
    in_file = InputMultiPath(File(exists=True), mandatory=True)
    entities = InputMultiPath(traits.Dict, usedefault=True,
                              desc='Per-file entities to include in filename')
    fixed_entities = traits.Dict(usedefault=True,
                                 desc='Entities to include in all filenames')
    path_patterns = InputMultiPath(
        traits.Str, desc='BIDS path patterns describing format of file names')


class BIDSDataSinkOutputSpec(TraitedSpec):
    out_file = OutputMultiPath(File, desc='output file')


class BIDSDataSink(IOBase):
    '''
    DataSink for producing moving several files to a nice BIDS Naming structure
    given files and a list of entities. All credit goes to Chris Markiewicz, Alejandro De La Vega,
    Dylan Nielson and Adina Wagner and the Fitlins team.
    '''
    input_spec = BIDSDataSinkInputSpec
    output_spec = BIDSDataSinkOutputSpec

    _always_run = True

    def _list_outputs(self):
        from bids.layout import BIDSLayout
        base_dir = self.inputs.base_directory

        os.makedirs(base_dir, exist_ok=True)

        layout = BIDSLayout(base_dir, validate=False)
        path_patterns = self.inputs.path_patterns
        if not isdefined(path_patterns):
            path_patterns = None

        out_files = []
        for entities, in_file in zip(self.inputs.entities,
                                     self.inputs.in_file):
            ents = {**self.inputs.fixed_entities}
            ents.update(entities)

            ents = {k: snake_to_camel(str(v)) for k, v in ents.items()}

            out_fname = os.path.join(
                base_dir, layout.build_path(
                    ents, path_patterns))
            os.makedirs(os.path.dirname(out_fname), exist_ok=True)

            _copy_or_convert(in_file, out_fname)
            out_files.append(out_fname)

        return {'out_file': out_files}

def _copy_or_convert(in_file, out_file):
    in_ext = bids_split_filename(in_file)[2]
    out_ext = bids_split_filename(out_file)[2]

    # Copy if filename matches
    if in_ext == out_ext:
        copyfile(in_file, out_file, copy=True, use_hardlink=True)
        return

    # gzip/gunzip if it's easy
    if in_ext == out_ext + '.gz' or in_ext + '.gz' == out_ext:
        read_open = GzipFile if in_ext.endswith('.gz') else open
        write_open = GzipFile if out_ext.endswith('.gz') else open
        with read_open(in_file, mode='rb') as in_fobj:
            with write_open(out_file, mode='wb') as out_fobj:
                shutil.copyfileobj(in_fobj, out_fobj)
        return

    # Let nibabel take a shot
    try:
        nb.save(nb.load(in_file), out_file)
    except Exception:
        pass
    else:
        return

    raise RuntimeError("Cannot convert {} to {}".format(in_ext, out_ext))
