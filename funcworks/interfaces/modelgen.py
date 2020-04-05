# pylint: disable=C0415,C0114,C0115,W0404,W0621,W0612
from pathlib import Path
from bids.layout.writing import build_path
from nipype.interfaces.base import (
    BaseInterfaceInputSpec, Bunch, TraitedSpec,
    InputMultiPath, OutputMultiPath, File,
    traits
    )
from nipype.interfaces.io import IOBase
import nibabel as nb
import numpy as np
from ..utils import snake_to_camel

class GetRunModelInfoInputSpec(BaseInterfaceInputSpec):
    functional_file = File(exists=True, mandatory=True)
    events_file = File(exists=True, mandatory=False)
    model = traits.Dict(mandatory=True)
    detrend_poly = traits.Any(default=None,
                              desc=('Legendre polynomials to regress out'
                                    'for temporal filtering'))

class GetRunModelInfoOutputSpec(TraitedSpec):
    run_info = traits.Any(desc='Model Info required to construct Run Level Model')
    event_regressors = traits.List(desc='List of event types included in Run Model')
    confound_regressors = traits.List(desc='List of confound_regressors included in Run Model')
    run_metadata = traits.Dict(desc='Metadata for current run')
    run_contrasts = traits.List(desc='List of tuples describing each contrasts')
    run_entities = traits.Dict(desc='Run specific BIDS Entities')
    contrast_entities = OutputMultiPath(traits.Dict(),
                                        desc='Contrast specific list of entities')
    motion_parameters = OutputMultiPath(File(exists=True),
                                        desc='File containing first six motion regressors')
    repetition_time = traits.Float(desc='Repetition Time for the dataset')
    contrast_names = traits.List(desc='List of Contrast Names to pass to higher levels')

class GetRunModelInfo(IOBase):
    '''Grabs EV files for subject based on contrasts of interest'''
    input_spec = GetRunModelInfoInputSpec
    output_spec = GetRunModelInfoOutputSpec

    #_always_run = True

    def _list_outputs(self):
        import json
        regressors_file, meta_file, entities = self._get_required_files()

        motion_params = self._get_motion_parameters(regressors_file=regressors_file)

        with open(meta_file, 'r') as meta_read:
            metadata = json.load(meta_read)
        entities.update({'Volumes': nb.load(self.inputs.functional_file).shape[3]})
        run_info, event_regs, confound_regs = self._get_model_info(
            regressors_file=regressors_file)
        run_conts, contrast_names = self._get_contrasts(event_names=event_regs)

        contrast_entities = self._get_entities(
            contrasts=run_conts, run_entities=entities)
        detrend_poly = self.inputs.detrend_poly
        entities.update({'Volumes': nb.load(self.inputs.functional_file).shape[3],
                         'DegreesOfFreedom' : len(event_regs + confound_regs)})

        if detrend_poly:
            polynomial_names, polynomial_arrays = \
                self._detrend_polynomial(regressors_file, detrend_poly)
            run_info.regressor_names.extend(polynomial_names) #pylint: disable=E1101
            run_info.regressors.extend(polynomial_arrays) #pylint: disable=E1101

        return {'run_info' : run_info,
                'event_regressors': event_regs,
                'confound_regressors': confound_regs,
                'run_metadata': metadata,
                'run_contrasts': run_conts,
                'motion_parameters': motion_params,
                'repetition_time': metadata['RepetitionTime'],
                'run_entities': entities,
                'contrast_entities': contrast_entities,
                'contrast_names': contrast_names}

    def _get_required_files(self):
        #A workaround to a current issue in pybids
        #that causes massive resource use when indexing derivative tsv files
        from pathlib import Path
        from bids.layout import parse_file_entities
        from bids.layout.writing import build_path
        func = Path(self.inputs.functional_file)
        entities = parse_file_entities(str(func))
        entities['run'] = '{:02d}'.format(entities['run'])
        entities.pop('suffix', None)
        confounds_pattern = \
        'sub-{subject}[_ses-{session}]_task-{task}_run-{run}_desc-confounds_regressors.tsv'
        meta_pattern = \
        'sub-{subject}[_ses-{session}]_task-{task}_run-{run}[_space-{space}]_desc-preproc_bold.json'
        regressors_file = func.parent / build_path(entities, path_patterns=confounds_pattern)
        meta_file = func.parent / build_path(entities, path_patterns=meta_pattern)
        return regressors_file, meta_file, entities

    def _get_model_info(self, regressors_file):
        import pandas as pd
        import numpy as np
        event_data = pd.read_csv(
            self.inputs.events_file, sep='\t')
        conf_data = pd.read_csv(
            regressors_file, sep='\t')
        conf_data.fillna(0, inplace=True)
        level_model = self.inputs.model
        events = []
        onsets = []
        amplitudes = []
        durations = []
        regressor_names = []
        regressors = []
        for regressor in level_model['Model']['X']:
            if '.' in regressor:
                event_column, event_name = regressor.split('.')
                event_frame = event_data.query(f'{event_column} == "{event_name}"')
                if event_frame.empty:
                    continue
                events.append(regressor)
                onsets.append(event_frame['onset'].values)
                durations.append(event_frame['duration'].values)
                amplitudes.append(np.ones(len(event_frame)))
            else:
                regressor_names.append(regressor)
                regressors.append(conf_data[regressor].values)

        run_info = Bunch(conditions=events,
                         onsets=onsets,
                         durations=durations,
                         amplitudes=amplitudes,
                         tmod=None,
                         pmod=None,
                         regressor_names=regressor_names,
                         regressors=regressors)

        return run_info, events, regressor_names

    def _get_contrasts(self, event_names):
        """
        Produces contrasts from a given model file and a run specific events file
        """
        import itertools as it
        model = self.inputs.model
        include_combos = list(it.combinations(event_names, 2))
        all_contrasts = []
        real_contrasts = model["Contrasts"]

        contrasts = []
        dummy_contrasts = []
        if 'Conditions' in model["DummyContrasts"]:
            dummy_contrasts = model["DummyContrasts"]['Conditions']
        else:
            dummy_contrasts = model["Model"]["X"]

        for dcontrast in dummy_contrasts:
            if dcontrast not in event_names and '.' in dcontrast:
                continue
            all_contrasts.append((dcontrast, 'T', [dcontrast], [1]))
            contrasts.append(dcontrast)

        for contrast in real_contrasts:
            if not any([all([x in contrast['ConditionList'], y in contrast['ConditionList']]) \
                        for x, y in include_combos])\
            and len(contrast['ConditionList']) == 2:
                continue
            contrasts.append(contrast)
            if contrast['Name'] == 'task_vs_baseline':
                weight_vector = [1 * 1 / len(event_names)] * len(event_names)
                all_contrasts.append((contrast['Name'], contrast['Type'].upper(),
                                      event_names,
                                      weight_vector))
            else:
                all_contrasts.append((contrast['Name'], contrast['Type'].upper(),
                                      contrast['ConditionList'],
                                      contrast['Weights']))
        return all_contrasts, contrasts

    @staticmethod
    def _get_motion_parameters(regressors_file):
        import os #pylint: disable=W0621,W0404
        import pandas as pd
        motion_params_path = os.path.join(
            os.getcwd(),
            os.path.basename(regressors_file).replace('regressors', 'motparams'))

        confound_data = pd.read_csv(regressors_file, sep='\t')
        #Motion data gets formatted FSL style, with x, y, z rotation, then x,y,z translation
        motion_data = confound_data[['rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z']]
        motion_data.to_csv(motion_params_path, sep='\t', header=None, index=None)
        motion_params = motion_params_path
        return motion_params

    @staticmethod
    def _get_entities(contrasts, run_entities):
        contrast_entities = []
        contrast_names = [contrast[0] for contrast in contrasts]
        for contrast_name in contrast_names:
            run_entities.update({'contrast':contrast_name})
            contrast_entities.append(run_entities.copy())
        return contrast_entities

    @staticmethod
    def _detrend_polynomial(regressors_file, detrend_poly=None):
        import numpy as np
        import pandas as pd
        from scipy.special import legendre

        regressors_frame = pd.read_csv(regressors_file)

        poly_names = []
        poly_arrays = []
        for i in range(0, detrend_poly + 1):
            poly_names.append(f'legendre{i:02d}')
            poly_arrays.append(legendre(i)(np.linspace(-1, 1, len(regressors_frame))))

        return poly_names, poly_arrays

class GenerateHigherInfoInputSpec(BaseInterfaceInputSpec):
    contrast_maps = InputMultiPath(File(exists=True),
                                   desc='List of contrasts statmaps from previous level')
    contrast_metadata = InputMultiPath(
        traits.Dict(desc='Contrast names inherited from previous levels'))
    model = traits.Dict(desc='Step level information from the model file')

class GenerateHigherInfoOutputSpec(TraitedSpec):
    effect_maps = traits.List()
    variance_maps = traits.List()
    dof_maps = traits.List()
    contrast_matrices = traits.List()
    design_matrices = traits.List()
    covariance_matrices = traits.List()
    contrast_metadata = traits.List()

class GenerateHigherInfo(IOBase):
    input_spec = GenerateHigherInfoInputSpec
    output_spec = GenerateHigherInfoOutputSpec

    _always_run = True

    def _list_outputs(self):
        organization, dummy_contrasts = self._get_organization()
        entities, effect_maps, variance_maps, dof_maps = \
            self._merge_maps(organization, dummy_contrasts)
        design_matrices, contrast_matrices, covariance_matrices = \
            self._produce_matrices(entities=entities)
        return {'effect_maps': effect_maps,
                'variance_maps': variance_maps,
                'dof_maps': dof_maps,
                'contrast_metadata': entities,
                'contrast_matrices': contrast_matrices,
                'design_matrices': design_matrices,
                'covariance_matrices': covariance_matrices}

    def _get_organization(self):
        model = self.inputs.model

        contrast_zip = zip(self.inputs.contrast_maps,
                           self.inputs.contrast_metadata)
        organization = {}
        #split_fields = []
        if "Transformations" in model:
            if model['Transformations']['Name'] == 'Split':
                split_fields = []
        for contrast_file, contrast_entities in contrast_zip:
            if contrast_entities['contrast'] not in organization:
                organization[contrast_entities['contrast']] = []
            organization[contrast_entities['contrast']].append(
                {'File': contrast_file, 'Metadata': contrast_entities})

        #for split_field in split_fields:
        #    pass
        dummy_contrasts = []
        if "DummyContrasts" in model:
            if 'Conditions' in model['DummyContrasts']:
                dummy_contrasts = model['DummyContrasts']['Conditions']
            else:
                dummy_contrasts = organization.keys()
        return organization, dummy_contrasts

    def _merge_maps(self, organization, dummy_contrasts):
        effect_maps = []
        variance_maps = []
        dof_maps = []
        contrast_metadata = []
        num_copes = []
        entities = []
        for dcontrast in dummy_contrasts:
            ceffect_maps = []
            cvariance_maps = []
            cdof_maps = []
            for bids_info in organization[dcontrast]:
                if 'stat' not in bids_info['Metadata']:
                    continue
                if bids_info['Metadata']['stat'] == 'effect':
                    open_file = nb.load(bids_info['File'])
                    affine = open_file.affine
                    dof_file = (np.ones_like(open_file.get_fdata())
                                * bids_info['Metadata']['DegreesOfFreedom'])
                    dof_file = nb.nifti1.Nifti1Image(dof_file, affine)
                    ceffect_maps.append(open_file)
                    cdof_maps.append(dof_file)
                elif bids_info['Metadata']['stat'] == 'variance':
                    open_file = nb.load(bids_info['File'])
                    cvariance_maps.append(open_file)

            ceffect_maps = nb.concat_images(ceffect_maps)
            cvariance_maps = nb.concat_images(cvariance_maps)
            cdof_maps = nb.concat_images(cdof_maps)
            centities = bids_info['Metadata'].copy()
            centities.pop('run', None)
            centities.pop('stat', None)
            centities.update({'numcopes': ceffect_maps.shape[-1]})
            entities.append(centities)
            ents = centities.copy()
            ents.update({'contrast': snake_to_camel(centities['contrast'])})

            merged_pattern = ('sub-{subject}[_ses-{session}]'
                              '_contrast-{contrast}_stat-{stat}'
                              '_desc-merged_statmap.nii.gz')
            ents['stat'] = 'effect'
            effects_path = Path.cwd() / build_path(ents, path_patterns=merged_pattern)
            nb.nifti1.save(ceffect_maps, effects_path)
            ents['stat'] = 'variance'
            variance_path = Path.cwd() / build_path(ents, path_patterns=merged_pattern)
            nb.nifti1.save(cvariance_maps, variance_path)
            ents['stat'] = 'dof'
            dof_path = Path.cwd() / build_path(ents, path_patterns=merged_pattern)
            nb.nifti1.save(cdof_maps, dof_path)
            effect_maps.append(str(effects_path))
            variance_maps.append(str(variance_path))
            dof_maps.append(str(dof_path))
        return entities, effect_maps, variance_maps, dof_maps


    def _produce_matrices(self, entities):
        design_matrices = []
        contrast_matrices = []
        covariance_matrices = []
        matrix_pattern = 'sub-{subject}[_ses-{session}]_contrast-{contrast}_desc-{desc}_design.mat'
        for entity in entities:
            ents = entity.copy()
            numcopes = ents['numcopes']
            ents['desc'] = 'contrast'
            contrast = ents['contrast']
            ents['contrast'] = snake_to_camel(entity['contrast'])

            conpath = Path.cwd() / build_path(ents, path_patterns=matrix_pattern)
            with open(conpath, 'a') as write_file:
                write_file.writelines(f'/ContrastName1 {contrast}\n')
                write_file.writelines(f'/NumWaves 1\n')
                write_file.writelines(f'/NumPoints 1\n\n')
                write_file.writelines('/Matrix\n')
                write_file.writelines('1\n')

            ents['desc'] = 'design'
            despath = Path.cwd() / build_path(ents, path_patterns=matrix_pattern)
            with open(despath, 'a') as write_file:
                write_file.writelines(f'/NumWaves 1\n')
                write_file.writelines(f'/NumPoints {numcopes}\n')
                write_file.writelines('/PPHeights 1\n\n')
                write_file.writelines('/Matrix\n')
                for _ in range(numcopes):
                    write_file.writelines('1\n')

            ents['desc'] = 'covariance'
            covpath = Path.cwd() / build_path(ents, path_patterns=matrix_pattern)
            with open(covpath, 'a') as write_file:
                write_file.writelines(f'/NumWaves 1\n')
                write_file.writelines(f'/NumPoints {numcopes}\n\n')
                write_file.writelines('/Matrix\n')
                for _ in range(numcopes):
                    write_file.writelines('1\n')
            design_matrices.append(str(despath))
            contrast_matrices.append(str(conpath))
            covariance_matrices.append(str(covpath))
        return design_matrices, contrast_matrices, covariance_matrices
