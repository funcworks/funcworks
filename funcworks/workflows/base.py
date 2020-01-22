import os
import json
from pathlib import Path
from copy import deepcopy
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from .fsl import fsl_first_level_wf, fsl_session_level_wf
#pylint: disable=R0913,R0914
def init_funcworks_wf(model,
                      bids_dir,
                      output_dir,
                      work_dir,
                      participants,
                      analysis_level,
                      smoothing,
                      derivatives,
                      run_uuid,
                      use_rapidart):
    with open(model, 'r') as read_mdl:
        model_dict = json.load(read_mdl)

    funcworks_wf = Workflow(name='funcworks_wf')
    funcworks_wf.base_dir = str(work_dir)
    if smoothing:
        smoothing_params = smoothing.split(':')
        if len(smoothing_params) == 1:
            smoothing_params.extend(('l1', 'iso'))
        elif len(smoothing_params) == 2:
            smoothing_params.append('iso')
        smoothing_fwhm, smoothing_level, smoothing_type = smoothing_params
        smoothing_fwhm = float(smoothing_fwhm)

        if smoothing_level.lower().startswith("l"):
            if int(smoothing_level[1:]) > len(model_dict['Steps']):
                raise ValueError(f"Invalid smoothing level {smoothing_level}")
    else:
        smoothing_fwhm = None
        smoothing_level = None
        smoothing_type = None

    for subject_id in participants:
        single_subject_wf = init_funcworks_single_subject_wf(model=model_dict,
                                                             bids_dir=bids_dir,
                                                             output_dir=output_dir,
                                                             work_dir=work_dir,
                                                             subject_id=subject_id,
                                                             analysis_level=analysis_level,
                                                             smoothing_fwhm=smoothing_fwhm,
                                                             smoothing_level=smoothing_level,
                                                             smoothing_type=smoothing_type,
                                                             derivatives=derivatives,
                                                             use_rapidart=use_rapidart,
                                                             name=f'single_subject_{subject_id}_wf')
        single_subject_wf.config['execution']['crashdump_dir'] = os.path.join(
            output_dir, "funcworks", "sub-" + subject_id, 'log', run_uuid)

        for node in single_subject_wf._get_all_nodes():
            node.config = deepcopy(single_subject_wf.config)

        funcworks_wf.add_nodes([single_subject_wf])

    return funcworks_wf

def init_funcworks_single_subject_wf(model,
                                     bids_dir,
                                     output_dir,
                                     work_dir,
                                     subject_id,
                                     analysis_level,
                                     smoothing_fwhm,
                                     smoothing_level,
                                     smoothing_type,
                                     derivatives,
                                     use_rapidart,
                                     name):

    funcworks_single_subject_wf = Workflow(name=name)
    run_model = None
    for step in model['Steps']:
        if not run_model:
            run_model = fsl_first_level_wf(model=model,
                                           step=step,
                                           bids_dir=bids_dir,
                                           output_dir=output_dir,
                                           work_dir=work_dir,
                                           subject_id=subject_id,
                                           smoothing_fwhm=smoothing_fwhm,
                                           smoothing_level=smoothing_level,
                                           smoothing_type=smoothing_type,
                                           derivatives=derivatives,
                                           use_rapidart=use_rapidart)
            funcworks_single_subject_wf.add_nodes([run_model])
        '''
        if not subject_model:
            subject_model = fsl_second_level_wf(model=model,
                                                step=step,
                                                bids_dir=bids_dir,
                                                output_dir=output_dir,
                                                work_dir=work_dir,
                                                subject_id=subject_id,
                                                smoothing=smoothing,
                                                derivatives=derivatives)
                                                '''

        if step == analysis_level:
            break
    return funcworks_single_subject_wf
