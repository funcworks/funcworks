# pylint: disable=C0415,C0114,C0115,W0404,W0621,W0612
from pathlib import Path
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

    _always_run = True

    def _list_outputs(self):
        import json
        regressors_file, meta_file, entities = self._get_required_files()

        motion_params = self._get_motion_parameters(regressors_file=regressors_file)

        with open(meta_file, 'r') as meta_read:
            metadata = json.load(meta_read)
        metadata.update({'NumTimepoints': nb.load(self.inputs.functional_file).shape[3]})
        run_info, event_regs, confound_regs = self._get_model_info(
            regressors_file=regressors_file)
        run_conts, contrast_names = self._get_contrasts(event_names=event_regs)

        contrast_entities = self._get_entities(
            contrasts=run_conts, run_entities=entities)
        detrend_poly = self.inputs.detrend_poly
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
    effects = InputMultiPath(desc='Effect Maps from earlier level')
    variances = InputMultiPath(desc='Variance Maps from earlier level')
    metadata = InputMultiPath(desc='Metadata Maps from earlier level')
    step_info = traits.Dict(desc='Step info from model file')
    contrast_names = traits.Dict(desc='Contrast names inherited from previous levels')

class GenerateHigherInfoOutputSpec(TraitedSpec):
    design_files = OutputMultiPath(File, desc='Design Matrix Files')
    contrast_files = OutputMultiPath(File, desc='Contrast Matrix Files')
    group_files = OutputMultiPath(File, desc=' Group Matrix Files')
    cope_files = OutputMultiPath(File, desc='Combined Contrast Parameter Files')
    variance_files = OutputMultiPath(File, desc='Combined Variance Files')
    dof_files = OutputMultiPath(File, desc='Combined DOF Files')

class GenerateHigherInfo(IOBase):
    inputspec = GenerateHigherInfoInputSpec
    outputspec = GenerateHigherInfoOutputSpec

    _always_run = True

    def _list_outputs(self):
        return self
        #return {
        #    'design_files': design_files,
        #    'contrast_files': contrast_files,
        #    'group_files': group_files,
        #    'cope_files': cope_files,
        #    'variance_files': variance_files,
        #    'dof_files': dof_files,
        #    'entities': entities}

    def _produce_design_files(self):
        from bids.layout import parse_file_entities
        step_info = self.inputs.step_info
        if "DummyContrasts" in step_info:
            if 'Conditions' in step_info['DummyContrasts']:
                dummy_contrasts = step_info['DummyContrasts']['Conditions']
            else:
                dummy_contrasts = self.inputs.contrast_names
        contrast_files = []
        cope_files = []
        variance_files = []
        dof_files = []
        contrast_entities = []
        step_info = self.inputs.step_info
        for contrast in dummy_contrasts:
            contrast_name = snake_to_camel(contrast)
            cont_effects = [x for x in self.inputs.effects if contrast_name in x]
            entities = parse_file_entities(cont_effects[0])
            entities.pop('run', None)
            cont_variance = [x for x in self.inputs.variances if contrast_name in x]
            affine = nb.load(cont_effects[0]).affine
            cont_effects = np.concatenate([nb.load(run) for run in cont_effects])
            cont_variance = np.concatenate([nb.load(run) for run in cont_variance])
            contrast_entities.append(entities.copy())
            design_files.append(design)
