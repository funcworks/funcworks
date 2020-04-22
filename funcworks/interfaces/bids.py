"""General BIDS interfaces."""
# pylint: disable=W0703,C0115,C0415
import json
import shutil
from pathlib import Path
from gzip import GzipFile
from nipype import logging
from nipype.utils.filemanip import copyfile
from nipype.interfaces.base import (
    BaseInterfaceInputSpec, TraitedSpec,
    InputMultiPath, OutputMultiPath, File, Directory,
    traits, isdefined, SimpleInterface)
from nipype.interfaces.io import IOBase
import nibabel as nb
from ..utils import snake_to_camel

iflogger = logging.getLogger("nipype.interface")


def bids_split_filename(fname):
    """
    Split a filename into parts: path, base filename, and extension.

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
        ".L.func.gii", ".L.func.gii",
        ".nii.gz", ".tsv.gz",
    ]
    file_path = Path(fname)
    pth = str(file_path.parent.as_posix())

    fname = str(file_path.name)
    for special_ext in special_extensions:
        if fname.lower().endswith(special_ext.lower()):
            ext = special_ext
            fname = fname[:-len(ext)]
            break
    else:
        fname = file_path.stem
        ext = file_path.suffix
    return pth, fname, ext


def _ensure_model(model):
    model = getattr(model, 'filename', model)

    if isinstance(model, str):
        if Path(model).is_file():
            with open(model) as fobj:
                model = json.load(fobj)
        else:
            model = json.loads(model)
    return model


class _BIDSDataSinkInputSpec(BaseInterfaceInputSpec):
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


class _BIDSDataSinkOutputSpec(TraitedSpec):
    out_file = OutputMultiPath(File, desc='output file')


class BIDSDataSink(IOBase):
    """
    Moves multiple files to a clean BIDS Naming Structure.

    DataSink for producing moving several files to a nice BIDS Naming structure
    given files and a list of entities. All credit goes to Chris Markiewicz,
    Alejandro De La Vega, Dylan Nielson and Adina Wagner and the Fitlins team.
    """

    input_spec = _BIDSDataSinkInputSpec
    output_spec = _BIDSDataSinkOutputSpec

    _always_run = True

    def _list_outputs(self):
        from bids.layout.writing import build_path
        base_dir = Path(self.inputs.base_directory)
        base_dir.mkdir(exist_ok=True, parents=True)  # pylint: disable=E1123

        path_patterns = self.inputs.path_patterns
        if not isdefined(path_patterns):
            path_patterns = None

        out_files = []
        for entities, in_file in zip(self.inputs.entities,
                                     self.inputs.in_file):
            ents = {**self.inputs.fixed_entities}
            ents.update(entities)

            ents = {k: snake_to_camel(str(v)) for k, v in ents.items()}

            out_fname = base_dir / build_path(ents, path_patterns)
            out_fname.parent.mkdir(exist_ok=True, parents=True)

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

    raise RuntimeError(f"Cannot convert {in_ext} to {out_ext}")


class _BIDSGetInputSpec(BaseInterfaceInputSpec):
    database_path = Directory(
        exists=True, mandatory=True, desc="Path to BIDS Dataset DBCACHE")
    fixed_entities = traits.Dict(
        desc="Queries for outfield outputs")
    align_volumes = traits.Either(
        traits.Int, None, default=None,
        desc='Run reference to align functional volumes')


class _BIDSGetOutputSpec(TraitedSpec):
    functional_files = OutputMultiPath(File)
    mask_files = OutputMultiPath(File)
    reference_files = OutputMultiPath(File)
    metadata_files = OutputMultiPath(File)
    events_files = OutputMultiPath(File)
    regressor_files = OutputMultiPath(File)
    entities = OutputMultiPath(traits.Dict)


class BIDSGet(SimpleInterface):
    """Interface that querys for functional files and associated masks/refs."""

    input_spec = _BIDSGetInputSpec
    output_spec = _BIDSGetOutputSpec
    _always_run = True
    _pkg = "bids"

    def _run_interface(self, runtime):
        from bids import BIDSLayout
        layout = BIDSLayout.load(database_path=self.inputs.database_path)
        fixed_entities = self.inputs.fixed_entities

        functional_entities = {
            **fixed_entities,
            'datatype': 'func', 'desc': 'preproc',
            'extension': 'nii.gz', 'suffix': 'bold'}
        functional_files = layout.get(**functional_entities)
        if len(functional_files) == 0:
            raise FileNotFoundError(
                f'Unable to find functional image with '
                f'specified entities {functional_entities}')

        outputs = dict(
            mask_files=[],
            reference_files=[],
            events_files=[],
            metadata_files=[],
            regressor_files=[],
            entities=[])

        for file in functional_files:
            ents = layout.parse_file_entities(file.path)
            if 'space' not in ents:
                ents['space'] = None
            file_ents = dict(
                events={
                    **ents,
                    'desc': None, 'extension': 'tsv',
                    'suffix': 'events', 'space': None},
                metadata={**ents, 'extension': 'json'},
                regressor={
                    **ents, 'desc': 'confounds', 'space': None,
                    'suffix': 'regressors', 'extension': 'tsv'},
                mask={**ents, 'desc': 'brain', 'suffix': 'mask'},
                reference={**ents, 'suffix': 'boldref', 'desc': None})
            if self.inputs.align_volumes and 'run' not in ents:
                raise ValueError(
                    f'Attempted to align to when run entity is not present in '
                    f'{file.path}.')
            elif self.inputs.align_volumes:
                file_ents['mask']['run'] = self.inputs.align_volumes
                file_ents['reference']['run'] = self.inputs.align_volumes

            ents.pop('suffix', None)
            ents.pop('desc', None)
            outputs['entities'].append(ents)
            for filetype, entities in file_ents.items():
                files = layout.get(**entities)
                if len(files) > 1:
                    raise ValueError(
                        f'More than one {filetype} produced for given '
                        f'entities {entities}\n'
                        f'{[x.path for x in files]}'
                        f'{ents}')
                elif len(files) == 0:
                    raise FileNotFoundError(
                        f'No {filetype} found for given entities '
                        f'{entities}')
                else:
                    outputs[f'{filetype}_files'].append(files[0])

        self._results['functional_files'] = functional_files
        self._results['mask_files'] = outputs['mask_files']
        self._results['reference_files'] = outputs['reference_files']
        self._results['metadata_files'] = outputs['metadata_files']
        self._results['events_files'] = outputs['events_files']
        self._results['regressor_files'] = outputs['regressor_files']
        self._results['entities'] = outputs['entities']

        return runtime
