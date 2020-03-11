#pylint: disable=R0913,R0914,C0114,C0116,W0212
import json
from pathlib import Path
from copy import deepcopy
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from .. import __version__
from .fsl import fsl_run_level_wf, fsl_higher_level_wf
flatten = lambda x: x[0]
def init_funcworks_wf(model_file,
                      bids_dir,
                      output_dir,
                      work_dir,
                      participants,
                      analysis_level,
                      smoothing,
                      derivatives,
                      run_uuid,
                      use_rapidart,
                      detrend_poly,
                      align_volumes):

    with open(model_file, 'r') as read_mdl:
        model = json.load(read_mdl)

    funcworks_wf = Workflow(name='funcworks_wf')
    (work_dir / model['Name']).mkdir(exist_ok=True, parents=True)
    funcworks_wf.base_dir = work_dir / model['Name']
    if smoothing:
        smoothing_params = smoothing.split(':')
        if len(smoothing_params) == 1:
            smoothing_params.extend(('l1', 'iso'))
        elif len(smoothing_params) == 2:
            smoothing_params.append('iso')
        smoothing_fwhm, smoothing_level, smoothing_type = smoothing_params
        smoothing_fwhm = float(smoothing_fwhm)

        if smoothing_level.lower().startswith("l"):
            if int(smoothing_level[1:]) > len(model['Steps']):
                raise ValueError(f"Invalid smoothing level {smoothing_level}")
    else:
        smoothing_fwhm = None
        smoothing_level = None
        smoothing_type = None

    for subject_id in participants:
        single_subject_wf = init_funcworks_subject_wf(model=model,
                                                      bids_dir=bids_dir,
                                                      output_dir=(output_dir /
                                                                  'funcworks' /
                                                                  model['Name']),
                                                      work_dir=work_dir,
                                                      subject_id=subject_id,
                                                      analysis_level=analysis_level,
                                                      smoothing_fwhm=smoothing_fwhm,
                                                      smoothing_level=smoothing_level,
                                                      smoothing_type=smoothing_type,
                                                      derivatives=derivatives,
                                                      use_rapidart=use_rapidart,
                                                      detrend_poly=detrend_poly,
                                                      align_volumes=align_volumes,
                                                      name=f'single_subject_{subject_id}_wf')
        crash_dir = (Path(output_dir) / 'funcworks' / 'logs' /
                     model['Name'] / f'sub-{subject_id}' / run_uuid)
        crash_dir.mkdir(exist_ok=True, parents=True)

        single_subject_wf.config['execution']['crashdump_dir'] = str(crash_dir)

        for node in single_subject_wf._get_all_nodes():
            node.config = deepcopy(single_subject_wf.config)

        funcworks_wf.add_nodes([single_subject_wf])

    return funcworks_wf

def init_funcworks_subject_wf(model,
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
                              detrend_poly,
                              align_volumes,
                              name):

    workflow = Workflow(name=name)
    #stage = None
    for step in model['Steps']:
        level = step['Level']
        if level == 'run':
            model = fsl_run_level_wf(model=model,
                                     step=step,
                                     bids_dir=bids_dir,
                                     output_dir=output_dir,
                                     work_dir=work_dir,
                                     subject_id=subject_id,
                                     smoothing_fwhm=smoothing_fwhm,
                                     smoothing_level=smoothing_level,
                                     smoothing_type=smoothing_type,
                                     derivatives=derivatives,
                                     use_rapidart=use_rapidart,
                                     detrend_poly=detrend_poly,
                                     align_volumes=align_volumes,
                                     name=f'fsl_run_level_wf')

        else:
            model = fsl_higher_level_wf(step=step,
                                        output_dir=output_dir,
                                        work_dir=work_dir,
                                        smoothing_fwhm=smoothing_fwhm,
                                        smoothing_level=smoothing_level,
                                        smoothing_type=smoothing_type,
                                        name=f'fsl_{level}_level_wf')

            workflow.connect([
                (stage, model,
                 [(f'collate_{pre_level}_outputs.out',
                   f'get_{level}_info.contrast_maps'),
                  (f'collate_{pre_level}_outputs.metadata',
                   f'get_{level}_info.contrast_metadata'),
                  ((f'bdg.brain_mask', flatten),
                   f'estimate_{level}_model.mask_file')])
            ])

        stage = model
        pre_level = level
        if level == analysis_level:
            break
    return workflow
