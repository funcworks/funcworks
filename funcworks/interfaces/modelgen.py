"""Interfaces for constructing models in FSL."""
from pathlib import Path
from nipype.interfaces.base import (
    BaseInterfaceInputSpec, Bunch, TraitedSpec,
    InputMultiPath, OutputMultiPath, File,
    traits, Directory)
from nipype.interfaces.io import IOBase
import nibabel as nb
import numpy as np
from ..utils import snake_to_camel


class _GetRunModelInfoInputSpec(BaseInterfaceInputSpec):
    metadata_file = File()
    regressor_file = File()
    events_file = File()
    entities = traits.Dict(mandatory=True)
    model = traits.Dict(mandatory=True)
    detrend_poly = traits.Any(
        default=None,
        desc=('Legendre polynomials to regress out'
              'for temporal filtering'))


class _GetRunModelInfoOutputSpec(TraitedSpec):
    run_info = traits.Any(
        desc='Model Info required to construct Run Level Model')
    run_contrasts = traits.List(
        desc='List of tuples describing each contrasts')
    contrast_entities = OutputMultiPath(
        traits.Dict(),
        desc='Contrast specific list of entities')
    motion_parameters = OutputMultiPath(
        File(exists=True),
        desc='File containing first six motion regressors')
    repetition_time = traits.Float(desc='Repetition Time for the dataset')
    contrast_names = traits.List(
        desc='List of Contrast Names to pass to higher levels')


class GetRunModelInfo(IOBase):
    """Grab files and information needed for an FSL run level model."""

    input_spec = _GetRunModelInfoInputSpec
    output_spec = _GetRunModelInfoOutputSpec
    # _always_run = True

    def _list_outputs(self):
        import json
        outputs = {}

        (outputs['motion_parameters'],
         n_timepoints) = self._get_motion_parameters()

        with open(self.inputs.metadata_file, 'r') as meta_read:
            run_metadata = json.load(meta_read)
        outputs['repetition_time'] = run_metadata['RepetitionTime']
        (outputs['run_info'], event_regressors,
         confound_regressors) = self._get_model_info()
        (outputs['run_contrasts'],
         outputs['contrast_names']) = self._get_contrasts(
             event_names=event_regressors)
        all_regressors = event_regressors + confound_regressors
        degrees_of_freedom = n_timepoints - len(all_regressors)
        outputs['contrast_entities'] = self._get_entities(
            contrasts=outputs['run_contrasts'],
            dof=degrees_of_freedom,
            n_timepoints=n_timepoints)

        if self.inputs.detrend_poly:
            polynomial_names, polynomial_arrays = self._detrend_polynomial()
            outputs['run_info'].regressor_names.extend(polynomial_names)
            outputs['run_info'].regressors.extend(polynomial_arrays)

        return outputs

    def _get_model_info(self):
        import pandas as pd
        import numpy as np

        level_model = self.inputs.model

        event_data = pd.read_csv(self.inputs.events_file, sep='\t')
        conf_data = pd.read_csv(self.inputs.regressor_file, sep='\t')
        conf_data.fillna(0, inplace=True)

        run_info = {'conditions': [],
                    'onsets': [],
                    'amplitudes': [],
                    'durations': [],
                    'regressor_names': [],
                    'regressors': []}
        for regressor in level_model['Model']['X']:
            if '.' in regressor:
                event_column, event_name = regressor.split('.')
                event_frame = event_data.query(
                    f'{event_column} == "{event_name}"')
                if event_frame.empty:
                    continue
                run_info['conditions'].append(regressor)
                run_info['onsets'].append(event_frame['onset'].values)
                run_info['durations'].append(event_frame['duration'].values)
                run_info['amplitudes'].append(np.ones(len(event_frame)))
            else:
                run_info['regressor_names'].append(regressor)
                run_info['regressors'].append(conf_data[regressor].values)

        run_info = Bunch(**run_info)
        return (run_info,
                run_info.conditions,  # pylint: disable=E1101
                run_info.regressor_names)  # pylint: disable=E1101

    def _get_contrasts(self, event_names):
        """Produce contrasts from a model step."""
        model = self.inputs.model
        contrast_spec = []
        real_contrasts = model["Contrasts"]
        contrast_names = []
        dummy_contrasts = []
        if 'Conditions' in model["DummyContrasts"]:
            dummy_contrasts = model["DummyContrasts"]['Conditions']
        else:
            dummy_contrasts = model["Model"]["X"]

        for dcontrast in dummy_contrasts:
            if dcontrast not in event_names and '.' in dcontrast:
                continue
            contrast_spec.append((dcontrast, 'T', [dcontrast], [1]))
            contrast_names.append(dcontrast)

        for contrast in real_contrasts:
            if not set(contrast['ConditionList']).issubset(set(event_names)):
                continue
            contrast_names.append(contrast['Name'])
            if contrast['Name'] == 'task_vs_baseline':
                weight_vector = [1 * 1 / len(event_names)] * len(event_names)
                contrast_spec.append((contrast['Name'],
                                      contrast['Type'].upper(),
                                      event_names,
                                      weight_vector))
            else:
                contrast_spec.append((contrast['Name'],
                                      contrast['Type'].upper(),
                                      contrast['ConditionList'],
                                      contrast['Weights']))
        return contrast_spec, contrast_names

    def _get_motion_parameters(self):
        from pathlib import Path
        import pandas as pd
        regressor_file = Path(self.inputs.regressor_file)
        motparams_path = str(regressor_file.name).replace(
            'regressors', 'motparams')
        motparams_path = Path.cwd() / motparams_path

        confound_data = pd.read_csv(regressor_file, sep='\t')
        n_timepoints = len(confound_data)
        # Motion data gets formatted FSL style, with x, y, z rotation,
        # then x,y,z translation
        motion_data = confound_data[['rot_x', 'rot_y', 'rot_z',
                                     'trans_x', 'trans_y', 'trans_z']]
        motion_data.to_csv(motparams_path, sep='\t', header=None, index=None)
        motion_params = motparams_path
        return motion_params, n_timepoints

    def _get_entities(self, contrasts, dof, n_timepoints):
        run_entities = self.inputs.entities.copy()
        contrast_entities = []
        contrast_names = [contrast[0] for contrast in contrasts]
        for contrast_name in contrast_names:
            run_entities['contrast'] = contrast_name
            run_entities['DegreesOfFreedom'] = dof
            run_entities['NumTimepoints'] = n_timepoints
            contrast_entities.append(run_entities.copy())
        return contrast_entities

    def _detrend_polynomial(self):
        import numpy as np
        import pandas as pd
        from scipy.special import legendre

        regressors_frame = pd.read_csv(self.inputs.regressor_file)

        poly_names = []
        poly_arrays = []
        for i in range(0, self.inputs.detrend_poly + 1):
            poly_names.append(f'legendre{i:02d}')
            poly_arrays.append(
                legendre(i)(np.linspace(-1, 1, len(regressors_frame))))

        return poly_names, poly_arrays


class _GenerateHigherInfoInputSpec(BaseInterfaceInputSpec):
    contrast_maps = InputMultiPath(
        File(exists=True), desc='List of statmaps from previous level')
    contrast_metadata = traits.List(
        desc='Contrast entities inherited from previous levels')
    model = traits.Dict(desc='Step level information from the model file')
    database_path = Directory(mandatory=True, exists=True)
    align_volumes = traits.Any(
        default=None,
        desc=('Target volume for functional realignment',
              'if not value is specified, will not functional file'))


class _GenerateHigherInfoOutputSpec(TraitedSpec):
    effect_maps = traits.List()
    variance_maps = traits.List()
    dof_maps = traits.List()
    contrast_matrices = traits.List()
    design_matrices = traits.List()
    covariance_matrices = traits.List()
    contrast_metadata = traits.List()
    brain_mask = traits.List()


class GenerateHigherInfo(IOBase):
    """Generate info for a level higher than first."""

    input_spec = _GenerateHigherInfoInputSpec
    output_spec = _GenerateHigherInfoOutputSpec

    _always_run = True

    def _list_outputs(self):
        from bids import BIDSLayout
        layout = BIDSLayout.load(self.inputs.database_path)
        organization = self._get_organization()
        (contrast_entities, effect_maps,
         variance_maps, dof_maps,
         brain_masks) = self._merge_maps(
             organization=organization, layout=layout)
        (design_matrices, contrast_matrices,
         covariance_matrices) = self._produce_matrices(
             contrast_entities=contrast_entities, layout=layout)
        return {'effect_maps': effect_maps,
                'variance_maps': variance_maps,
                'dof_maps': dof_maps,
                'contrast_metadata': contrast_entities,
                'contrast_matrices': contrast_matrices,
                'design_matrices': design_matrices,
                'covariance_matrices': covariance_matrices,
                'brain_mask': brain_masks}

    def _get_organization(self):
        model = self.inputs.model
        contrast_zip = zip(self.inputs.contrast_maps,
                           self.inputs.contrast_metadata)
        organization = {}
        split_fields = ['contrast', 'space', 'stat']
        if "Transformations" in model:
            for transform in model['Transformations']:
                if transform['Name'] == 'Split':
                    split_fields.append(transform['By'])
        for contrast_file, contrast_ents in contrast_zip:
            if contrast_ents['stat'] not in ['effect', 'variance']:
                continue
            fields = [f'{{{field}}}' for field in split_fields]
            if 'space' not in contrast_ents.keys():
                contrast_ents['space'] = 'bold'
            org_key = '.'.join(fields).format(**contrast_ents)
            if contrast_ents['space'] == 'bold':
                contrast_ents['space'] = None
            degrees_of_freedom = contrast_ents.pop('DegreesOfFreedom', None)
            if org_key not in organization.keys():
                organization[org_key] = {'Files': [contrast_file]}
                organization[org_key]['Metadata'] = contrast_ents.copy()
                organization[org_key]['Metadata']['DegreesOfFreedom'] = [
                    degrees_of_freedom]
            else:
                organization[org_key]['Files'].append(contrast_file)
                organization[org_key]['Metadata']['DegreesOfFreedom'].append(
                    degrees_of_freedom)
        for org_key in organization:
            organization[org_key]['Metadata']['NumLevelTimepoints'] = len(
                organization[org_key]['Files'])
            organization[org_key]['Metadata'].pop('stat', None)

        return organization

    def _merge_maps(self, organization, layout):
        merged_patt = ('sub-{subject}_[ses-{session}_][space-{space}_]'
                       'contrast-{contrast}_stat-{stat}_'
                       'desc-merged_statmap.nii.gz')
        maps_info = {'effect_maps': [],
                     'dof_maps': [],
                     'variance_maps': [],
                     'map_entities': [],
                     'mask_files': []}
        for org in organization:
            metadata = organization[org]['Metadata']
            org_files = organization[org]['Files']
            merged_image = nb.concat_images(
                [nb.load(file) for file in sorted(org_files)])

            if 'effect' in org:
                dof_data = np.ones_like(merged_image.get_fdata())
                dofs = metadata.pop('DegreesOfFreedom')

                if self.inputs.align_volumes and 'run' not in metadata:
                    raise ValueError(
                        'align_volumes is specified, but dataset '
                        'does not appear to contain multiple runs')
                elif self.inputs.align_volumes:
                    align_volume = self.inputs.align_volumes
                    mask_entities = {**metadata, 'desc': 'brain',
                                     'suffix': 'mask', 'run': align_volume}
                else:
                    mask_entities = {**metadata, 'desc': 'brain',
                                     'suffix': 'mask'}
                mask_path = layout.get(**mask_entities)
                if len(mask_path) > 1:
                    raise ValueError('Entities given produced '
                                     'more than one mask file')
                if isinstance(mask_path, list):
                    mask_path = str(Path(mask_path[0].path).as_posix())
                maps_info['mask_files'].append(mask_path)

                if metadata['space'] is None:
                    metadata.pop('space', None)
                    metadata.pop('run', None)
                metadata.pop('stat', None)
                maps_info['map_entities'].append(metadata.copy())
                metadata['contrast'] = snake_to_camel(metadata['contrast'])

                stat_name = 'dof'
                for i, dof in enumerate(dofs):
                    dof_data[:, :, :, i] *= float(dof)
                dof_path = layout.build_path(
                    {**metadata, 'stat': stat_name},
                    path_patterns=merged_patt, validate=False)
                dof_path = str((Path.cwd() / dof_path).as_posix())
                dof_image = nb.nifti1.Nifti1Image(
                    dof_data, merged_image.affine)
                maps_info['dof_maps'].append(dof_path)
                nb.nifti1.save(dof_image, dof_path)

                stat_name = 'effect'
            if 'variance' in org:
                stat_name = 'variance'
            merged_path = layout.build_path(
                {**metadata,
                 'stat': stat_name,
                 'desc': 'merged',
                 'suffix': 'statmap',
                 'contrast': snake_to_camel(metadata['contrast'])},
                path_patterns=merged_patt,
                validate=False)
            merged_path = str((Path.cwd() / merged_path).as_posix())
            maps_info[f'{stat_name}_maps'].append(merged_path)
            nb.nifti1.save(merged_image, merged_path)

        return (maps_info['map_entities'], maps_info['effect_maps'],
                maps_info['variance_maps'], maps_info['dof_maps'],
                maps_info['mask_files'])

    def _produce_matrices(self, contrast_entities, layout):

        matrix_paths = {'design_matrices': [],
                        'contrast_matrices': [],
                        'covariance_matrices': []}

        header_lines = {'contrast': ['/ContrastName1 {contrast}\n',
                                     '/NumWaves 1\n',
                                     '/NumPoints 1\n\n',
                                     '/Matrix\n',
                                     '1\n'],
                        'design': ['/NumWaves 1\n',
                                   '/NumPoints {numcopes}\n',
                                   '/PPHeights 1\n\n',
                                   '/Matrix\n'],
                        'covariance': ['/NumWaves 1\n',
                                       '/NumPoints {numcopes}\n\n',
                                       '/Matrix\n']}
        matrix_patt = ('sub-{subject}_[ses-{session}_]'
                       'contrast-{contrast}_desc-{desc}_matrix.mat')
        for entity in contrast_entities:
            ents = entity.copy()
            ents['contrast'] = snake_to_camel(ents['contrast'])
            for matrix_type in ['design', 'contrast', 'covariance']:
                matrix_path = layout.build_path(
                    {**ents, 'desc': matrix_type},
                    path_patterns=matrix_patt, validate=False)
                matrix_path = Path.cwd() / matrix_path
                if matrix_path.is_file():  # Remove file if it exists
                    matrix_path.unlink()
                matrix_path = str(matrix_path.as_posix())
                mat_file = open(matrix_path, 'a')
                for header_line in header_lines[matrix_type]:
                    mat_file.writelines(header_line.format(
                        contrast=entity['contrast'],
                        numcopes=ents['NumLevelTimepoints']))
                if matrix_type != 'contrast':
                    for _ in range(ents['NumLevelTimepoints']):
                        mat_file.writelines('1\n')
                mat_file.close()
                matrix_paths[f'{matrix_type}_matrices'].append(matrix_path)
        return (matrix_paths['design_matrices'],
                matrix_paths['contrast_matrices'],
                matrix_paths['covariance_matrices'])
