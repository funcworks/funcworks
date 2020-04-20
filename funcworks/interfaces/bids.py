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
    InputMultiPath, OutputMultiPath, File, Directory, Str, Undefined,
    traits, isdefined, LibraryBaseInterface, DynamicTraitedSpec)
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


class _BIDSDataGrabberInputSpec(DynamicTraitedSpec):
    database_path = Directory(
        exists=True, mandatory=True, desc="Path to BIDS Dataset DBCACHE")
    output_query = traits.Dict(
        key_trait=Str,
        value_trait=traits.Dict, desc="Queries for outfield outputs")
    raise_on_empty = traits.Bool(
        True,
        usedefault=True,
        desc="Generate exception if list is empty for a given field")


class _BIDSDataGrabberOutputSpec(DynamicTraitedSpec):
    bold_files = OutputMultiPath(File)


class BIDSDataGrabber(LibraryBaseInterface, IOBase):
    """
    Module that allows arbitrary query of BIDS Datasets.

    Examples
    --------
    By default, the BIDSDataGrabber fetches anatomical and functional images
    from a project, and makes BIDS entities (e.g. subject) available for
    filtering outputs.
    >>> bg = BIDSDataGrabber()
    >>> bg.inputs.base_dir = 'ds005/'
    >>> bg.inputs.subject = '01'
    >>> results = bg.run() # doctest: +SKIP
    Dynamically created, user-defined output fields can also be defined to
    return different types of outputs from the same project. All outputs
    are filtered on common entities, which can be explicitly defined as
    infields.
    >>> bg = BIDSDataGrabber(infields = ['subject'])
    >>> bg.inputs.base_dir = 'ds005/'
    >>> bg.inputs.subject = '01'
    >>> bg.inputs.output_query['dwi'] = dict(datatype='dwi')
    >>> results = bg.run() # doctest: +SKIP
    """

    input_spec = _BIDSDataGrabberInputSpec
    output_spec = _BIDSDataGrabberOutputSpec
    _always_run = False
    _pkg = "bids"

    def __init__(self, infields=None, **kwargs):
        """See help(BIDSDataGrabber) for more info."""
        super(BIDSDataGrabber, self).__init__(**kwargs)

        if not isdefined(self.inputs.output_query):
            self.inputs.output_query = {
                "bold": {
                    "datatype": "func",
                    "suffix": "bold",
                    "extensions": ["nii", ".nii.gz"],
                },
                "T1w": {
                    "datatype": "anat",
                    "suffix": "T1w",
                    "extensions": ["nii", ".nii.gz"],
                },
            }

        # If infields is empty, use all BIDS entities
        if infields is None:
            from bids import layout as bidslayout

            bids_config = (
                Path(bidslayout.__file__).parent
                / "config" / "bids.json")
            bids_config = json.load(open(bids_config, "r"))
            infields = [i["name"] for i in bids_config["entities"]]

        self._infields = infields or []

        # used for mandatory inputs check
        undefined_traits = {}
        for key in self._infields:
            self.inputs.add_trait(key, traits.Any)
            undefined_traits[key] = kwargs[key] if key in kwargs else Undefined

        self.inputs.trait_set(trait_change_notify=False, **undefined_traits)

    def _list_outputs(self):
        from bids import BIDSLayout

        layout = BIDSLayout.load(self.inputs.database_path)

        # If infield is not given nm input value, silently ignore
        filters = {}
        for key in self._infields:
            value = getattr(self.inputs, key)
            if isdefined(value):
                filters[key] = value

        outputs = {}
        for key, query in self.inputs.output_query.items():
            args = query.copy()
            args.update(filters)
            filelist = layout.get(return_type="file", **args)
            if len(filelist) == 0:
                msg = "Output key: %s returned no files" % key
                if self.inputs.raise_on_empty:
                    raise IOError(msg)
                iflogger.warning(msg)
                filelist = Undefined

            outputs[key] = filelist

        return outputs['bold_files']


def add_traits(base, names, trait_type=None):
    """
    Add traits to a traited class.

    All traits are set to Undefined by default
    """
    if trait_type is None:
        trait_type = traits.Any
    undefined_traits = {}
    for key in names:
        base.add_trait(key, trait_type)
        undefined_traits[key] = Undefined
    base.trait_set(trait_change_notify=False, **undefined_traits)
    # access each trait
    for key in names:
        _ = getattr(base, key)
    return base
