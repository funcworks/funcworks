import os
import shutil
from nipype.interfaces.base import (
    BaseInterface, BaseInterfaceInputSpec, Bunch, TraitedSpec, SimpleInterface,
    InputMultiPath, OutputMultiPath, File, Directory,
    traits, isdefined
    )
from nipype.interfaces.io import IOBase
import nibabel as nb

class GetModelInfoInputSpec(BaseInterfaceInputSpec):
    functional_file = File(exists=True, mandatory=True)
    events_file = File(exists=True, mandatory=False)
    model = traits.Dict(mandatory=True)
    detrend_poly = traits.Any(default=None,
                              desc=('Legendre polynomials to regress out'
                                    'for temporal filtering'))

class GetModelInfoOutputSpec(TraitedSpec):
    run_info = traits.Any(desc='Model Info required to construct Run Level Model')
    event_regressors = traits.List(desc='List of event types included in Run Model')
    confound_regressors = traits.List(desc='List of confound_regressors included in Run Model')
    run_metadata = traits.Dict(desc='Metadata for current run')
    run_contrasts = traits.List(desc='List of tuples describing each contrasts')
    motion_parameters = OutputMultiPath(File(exists=True),
                                        desc='File containing first six motion regressors')
    repetition_time = traits.Float(desc='Repetition Time for the dataset')

class GetModelInfo(IOBase):
    '''Grabs EV files for subject based on contrasts of interest'''
    input_spec = GetModelInfoInputSpec
    output_spec = GetModelInfoOutputSpec

    _always_run = True

    def _list_outputs(self):
        import json
        regressors_file, meta_file = self._get_required_files()

        motion_params = self._get_motion_parameters(regressors_file=regressors_file)

        level_model = self.inputs.model
        event_names = level_model['DummyContrasts']['Conditions']
        confound_names = [var for var in level_model['Model']['X'] if var not in event_names]

        with open(meta_file, 'r') as meta_read:
            metadata = json.load(meta_read)
        metadata.update({'NumTimepoints': nb.load(self.inputs.functional_file).shape[3]})
        run_info, event_regs, confound_regs = \
            self._get_model_info(regressors_file=regressors_file,
                                 events_file=self.inputs.events_file,
                                 confound_names=confound_names,
                                 event_names=event_names)
        run_conts = self._get_contrasts(model=level_model,
                                        event_names=event_regs)

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
                'repetition_time': metadata['RepetitionTime']}

    def _get_required_files(self):
        #A workaround to a current issue in pybids
        #that causes massive resource use when indexing derivative tsv files
        import os.path as op
        from bids.layout.writing import build_path
        func = self.inputs.functional_file
        entities = {pair.split('-')[0]:pair.split('-')[1] \
                    for pair in op.basename(func).split('_') if '-' in pair}
        confounds_pattern = \
        'sub-{sub}[_ses-{ses}]_task-{task}_run-{run}_desc-confounds_regressors.tsv'
        meta_pattern = \
        'sub-{sub}[_ses-{ses}]_task-{task}_run-{run}[_space-{space}]_desc-preproc_bold.json'
        regressors_file = op.join(op.dirname(func),
                                  build_path(entities, path_patterns=confounds_pattern))
        meta_file = op.join(op.dirname(func),
                            build_path(entities, path_patterns=meta_pattern))
        return regressors_file, meta_file

    @staticmethod
    def _get_model_info(regressors_file, events_file, confound_names, event_names):

        import pandas as pd
        import numpy as np
        event_data = pd.read_csv(events_file, sep='\t')
        conf_data = pd.read_csv(regressors_file, sep='\t')
        conf_data.fillna(0, inplace=True)
        events = []
        onsets = []
        amplitudes = []
        durations = []
        regressor_names = []
        regressors = []
        for event in event_names:
            if '.' in event:
                event_column, event_name = event.split('.')
                event_frame = event_data.query(f'{event_column} == "{event_name}"')
            else:
                event_frame = event_data.query(f'trial_type == {event}')
            if not event_frame.empty:
                events.append(event)
                onsets.append(event_frame['onset'].values)
                durations.append(event_frame['duration'].values)
                amplitudes.append(np.ones(len(event_frame)))

        for confound in confound_names:
            regressor_names.append(confound)
            regressors.append(conf_data[confound].values)

        run_info = Bunch(conditions=events,
                         onsets=onsets,
                         durations=durations,
                         amplitudes=amplitudes,
                         tmod=None,
                         pmod=None,
                         regressor_names=confound_names,
                         regressors=regressors)
        return run_info, events, regressor_names

    @staticmethod
    def _get_contrasts(model, event_names):
        """
        Produces contrasts from a given model file and a run specific events file
        """
        import itertools as it
        include_combos = list(it.combinations(event_names, 2))
        all_contrasts = []
        contrasts = model["Contrasts"]
        dummy_contrasts = model["DummyContrasts"]['Conditions']
        for dcontrast in dummy_contrasts:
            if dcontrast not in event_names:
                continue
            all_contrasts.append((dcontrast,
                                  'T',
                                  [dcontrast],
                                  [1]))
        for contrast in contrasts:
            if not any([all([x in contrast['ConditionList'], y in contrast['ConditionList']]) \
                        for x, y in include_combos])\
            and len(contrast['ConditionList']) == 2:
                continue

            if contrast['Name'] == 'task_vs_baseline':
                weight_vector = [1 * 1 / len(event_names)] * len(event_names)
                all_contrasts.append((contrast['Name'], contrast['Type'].upper(),
                                      event_names,
                                      weight_vector))
            else:
                all_contrasts.append((contrast['Name'], contrast['Type'].upper(),
                                      contrast['ConditionList'],
                                      contrast['Weights']))
        return all_contrasts

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
