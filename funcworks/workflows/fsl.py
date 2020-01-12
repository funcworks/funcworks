import os
import sys
import json
from pathlib import Path
import nibabel as nb
from nipype.pipeline import engine as pe
from nipype.interfaces import fsl, afni, base, io
from nipype.interfaces.utility import Function, IdentityInterface
from nipype.algorithms import modelgen, rapidart as ra
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

def fsl_first_level_wf(model,
                       step,
                       bids_dir, 
                       output_dir,
                       work_dir,
                       subject_id,
                       derivatives,
                       smoothing_fwhm=None,
                       smoothing_level=None,
                       smoothing_type=None,
                       use_rapidart=False,
                       name='fsl_first_level_wf'):
    """
    This workflow generates processes function data with information given in
    the model file

    model file contents
    --------------------
    contrasts = a dictionary containing contrast names as keys, with each value containing information for FSL Featmodel
                an example being {'contrast1$contrast2' : 'T'} where contrast1 and contrast2 are trial_types from the events.tsv
    confounds = a list of confounds to use in regression model

    Parameters
    ----------
    omp_nthreads

    """
    pop_lambda = lambda x: x[0]

    design_matrix_pattern = '[sub-{subject}/][ses-{session}/]' \
        '[sub-{subject}_][ses-{session}_]task-{task}[_acq-{acquisition}]' \
        '[_rec-{reconstruction}][_run-{run}][_echo-{echo}]_{suffix<design>}.tsv'
    contrast_pattern = '[sub-{subject}/][ses-{session}/]' \
        '[sub-{subject}_][ses-{session}_]task-{task}[_acq-{acquisition}]' \
        '[_rec-{reconstruction}][_run-{run}][_echo-{echo}][_space-{space}]_' \
        'contrast-{contrast}_stat-{stat<effect|variance|z|p|t|F>}_statmap.nii.gz'
    bids_dir = Path(bids_dir)
    work_dir = Path(work_dir)
    workflow = pe.Workflow(name=name)
            
    workflow.__desc__ = ""
    workflow.base_dir = work_dir / model['Name']
    confound_names = \
    [x for x in step['Model']['X'] if x not in step['DummyContrasts']['Conditions']]
    subject_source = pe.Node(IdentityInterface(["subject"]),
                             name="subject_source")
    
    bdg = pe.Node(io.BIDSDataGrabber(), name='bdg')
    bdg.inputs.base_dir = bids_dir
    bdg.inputs.subject = subject_id
    bdg.inputs.extra_derivatives = derivatives
    bdg.inputs.index_derivatives = True
    bdg.inputs.output_query = {'func': {'datatype':'func', 'desc':'preproc', 'extension':'nii.gz', 
                                    'suffix':'bold', 'task': model['Input']['task'], 'space':None},
                               'events': {'datatype':'func', 'suffix':'events', 'extension':'tsv',
                                          'task': model['Input']['task']},
                               'brain_mask': {'datatype': 'func', 'desc': 'brain',  'extension': 'nii.gz',
                                              'suffix':'mask', 'task': model['Input']['task'],
                                              'space':None}}
    
    exec_get_metadata = pe.MapNode(Function(input_names=['func'],
                                            output_names=['repetition_time', 'num_timepoints'],
                                            function=get_metadata),
                                   iterfield=['func'],
                                   name='exec_get_metadata')
    
    exec_get_confounds = pe.MapNode(Function(input_names=['func'],
                                             output_names=['confounds_file'],
                                             function=get_confounds),
                                   iterfield=['func'],
                                   name='exec_get_confounds')
    
    apply_brainmask = pe.MapNode(fsl.ImageMaths(suffix = '_bet',
                                                op_string = '-mas'),
                                 iterfield=['in_file', 'in_file2'],
                                 name='apply_brainmask')
        
    exec_get_info = pe.MapNode(Function(input_names=['func', 'events', 'confounds', 'confound_regressors'],
                                        output_names=['output', 'names'],
                                        function=get_info),
                            iterfield=['func', 'events', 'confounds'],
                            name='exec_get_info')
    
    exec_get_contrasts = pe.MapNode(Function(input_names=['step', 'include_contrasts'],
                                             output_names=['contrasts'],
                                             function=get_contrasts),
                                    iterfield=['include_contrasts'],
                                    name='exec_get_contrasts')
    exec_get_contrasts.inputs.step = step

    specify_model = pe.MapNode(modelgen.SpecifyModel(), iterfield=['functional_runs', 'subject_info', 'time_repetition'],
                               name='specify_model')
    specify_model.inputs.high_pass_filter_cutoff = -1.0
    specify_model.inputs.input_units = 'secs'
    
    fit_model = pe.MapNode(IdentityInterface(fields=['session_info', 'interscan_interval',
                                                     'contrasts', 'film_threshold',
                                                     'functional_data', 'bases',
                                                     'model_serial_correlations'],
                                             mandatory_inputs=True),
                           iterfield=['functional_data', 'session_info', 'interscan_interval', 'contrasts'],
                           name='fit_model')
    fit_model.inputs.bases = {'gamma':{'derivs': False}}
    fit_model.inputs.film_threshold = 0.0
    fit_model.inputs.model_serial_correlations = True
    
    first_level_design = pe.MapNode(fsl.Level1Design(),
                                    iterfield=['session_info', 'interscan_interval', 'contrasts'],
                                    name='first_level_design')
 
    generate_model = pe.MapNode(fsl.FEATModel(),
                                iterfield=['fsf_file', 'ev_files'],
                                name='generate_model')
    generate_model.inputs.environ = {'FSLOUTPUTTYPE': 'NIFTI_GZ'}
    generate_model.inputs.output_type = 'NIFTI_GZ'
    
    estimate_model = pe.MapNode(fsl.FILMGLS(),
                                iterfield=['design_file', 'in_file', 'tcon_file'],
                                name='estimate_model')
    estimate_model.inputs.environ = {'FSLOUTPUTTYPE': 'NIFTI_GZ'}
    estimate_model.inputs.mask_size = 5
    estimate_model.inputs.output_type = 'NIFTI_GZ'
    estimate_model.inputs.results_dir = 'results'
    estimate_model.inputs.smooth_autocorr = True
    
    outputnode = pe.MapNode(Function(input_names=['bids_dir', 'output_dir', 'contrast_pattern', 'contrasts', 'entities',
                                                  'effects', 'variances', 'z', 't', 'dof'],
                                     output_names=['effects', 'variances', 'z', 'p', 't', 'dof', 'F'],
                                     function=rename_outputs), 
                            iterfield=['entities', 'effects', 'variances', 'z', 't', 'dof', 'contrasts'],
                            name='outputnode')
    outputnode.inputs.bids_dir = bids_dir
    outputnode.inputs.output_dir = output_dir
    outputnode.inputs.contrast_pattern = contrast_pattern
    
    #Setup connections among workflow nodes
    workflow.connect(bdg, 'func', apply_brainmask, 'in_file')
    workflow.connect(bdg, 'brain_mask', apply_brainmask, 'in_file2')

    workflow.connect(bdg, 'func', exec_get_metadata, 'func')
    
    workflow.connect(bdg, 'func', exec_get_confounds, 'func')
    
    if use_rapidart:
        exec_get_motion_parameters = pe.MapNode(Function(input_names=['confounds'],
                                                         output_names='motion_params',
                                                         function=get_motion_parameters),
                                                iterfield=['confounds'],
                                                name='exec_get_motion_parameters')
        
        run_rapidart = pe.MapNode(ra.ArtifactDetect(use_differences=[True, False],
                                                    use_norm=True,
                                                    zintensity_threshold=3,
                                                    norm_threshold=1,
                                                    bound_by_brainmask=True,
                                                    mask_type='file',
                                                    parameter_source='FSL'),
                                  iterfield=['realignment_parameters', 
                                             'realigned_files', 
                                             'mask_file'],
                                  name='run_rapidart')
        
        reshape_rapidart = pe.MapNode(Function(input_names=['outlier_files', 'confounds', 'confound_regressors', 'num_timepoints'],
                                               output_names=['confounds', 'confound_regressors'],
                                               function=reshape_ra),
                                      iterfield=['outlier_files', 'confounds', 'num_timepoints'],
                                      name='reshape_rapidart')
        reshape_rapidart.inputs.confound_regressors = confound_names
        
        workflow.connect(exec_get_confounds, 'confounds_file', exec_get_motion_parameters, 'confounds')
        
        workflow.connect(exec_get_motion_parameters, 'motion_params', run_rapidart, 'realignment_parameters')
        workflow.connect(bdg, 'func', run_rapidart, 'realigned_files')
        workflow.connect(bdg, 'brain_mask', run_rapidart, 'mask_file')
        
        workflow.connect(run_rapidart, 'outlier_files', reshape_rapidart, 'outlier_files')
        workflow.connect(exec_get_confounds, 'confounds_file', reshape_rapidart, 'confounds')
        workflow.connect(exec_get_metadata, 'num_timepoints', reshape_rapidart, 'num_timepoints')
        
        exec_get_info.iterfield = ['func', 'events', 'confounds', 'confound_regressors']
        
        workflow.connect(reshape_rapidart, 'confounds', exec_get_info, 'confounds')
        workflow.connect(reshape_rapidart, 'confound_regressors', exec_get_info, 'confound_regressors')
        
    else:
        workflow.connect(exec_get_confounds, 'confounds_file', exec_get_info, 'confounds')
        exec_get_info.inputs.confound_regressors = confound_names 

                                               
    workflow.connect(bdg, 'func', exec_get_info, 'func')
    workflow.connect(bdg, 'events', exec_get_info, 'events')

    if smoothing_level == 'l1':
        setup_susan = pe.MapNode(Function(input_names=['func', 'brain_mask'],
                                          output_names=['usans', 'brightness_threshold'],
                                          function=get_smoothing_info_fsl),
                                 iterfield=['func', 'brain_mask'],
                                 name='setup_susan')
        
        run_susan = pe.MapNode(fsl.SUSAN(), 
                               iterfield=['in_file', 'brightness_threshold', 'usans'],
                               name='run_susan')
        run_susan.inputs.fwhm = smoothing_fwhm
        
        apply_mask_smooth = pe.MapNode(fsl.ImageMaths(suffix='_bet',
                                                      op_string='-mas'),
                                       iterfield=['in_file', 'in_file2'],
                                       name='apply_mask_smooth')

        workflow.connect(apply_brainmask, 'out_file', setup_susan, 'func')
        workflow.connect(bdg, 'brain_mask', setup_susan, 'brain_mask')

        workflow.connect(apply_brainmask, 'out_file', run_susan, 'in_file')
        workflow.connect(setup_susan, 'brightness_threshold', run_susan, 'brightness_threshold')
        workflow.connect(setup_susan, 'usans', run_susan, 'usans')
        
        workflow.connect(run_susan, 'smoothed_file', apply_mask_smooth, 'in_file')
        workflow.connect(bdg, 'brain_mask', apply_mask_smooth, 'in_file2')
        
        workflow.connect(apply_mask_smooth, 'out_file', specify_model, 'functional_runs')
        workflow.connect(apply_mask_smooth, 'out_file', fit_model, 'functional_data')
    else:
        workflow.connect(apply_brainmask, 'out_file', specify_model, 'functional_runs')
        workflow.connect(apply_brainmask, 'out_file', fit_model, 'functional_data')
        
    workflow.connect(exec_get_info, 'output', specify_model, 'subject_info')
    workflow.connect(exec_get_metadata, 'repetition_time', specify_model, 'time_repetition')
    
    workflow.connect(exec_get_info, 'names', exec_get_contrasts, 'include_contrasts')
        
    workflow.connect(specify_model, 'session_info', fit_model, 'session_info')
    workflow.connect(exec_get_metadata, 'repetition_time', fit_model, 'interscan_interval')
    workflow.connect(exec_get_contrasts, 'contrasts', fit_model, 'contrasts')
    
    workflow.connect(fit_model, 'interscan_interval',
                     first_level_design, 'interscan_interval')
    workflow.connect(fit_model, 'session_info', first_level_design, 'session_info')
    workflow.connect(fit_model, 'contrasts', first_level_design, 'contrasts')
    workflow.connect(fit_model, 'bases', first_level_design, 'bases')
    workflow.connect(fit_model, 'model_serial_correlations',
                     first_level_design, 'model_serial_correlations')
    
    workflow.connect(first_level_design, 'fsf_files', generate_model, 'fsf_file')
    workflow.connect(first_level_design, 'ev_files', generate_model, 'ev_files')
    
    workflow.connect(fit_model, 'film_threshold', estimate_model, 'threshold')
    workflow.connect(fit_model, 'functional_data', estimate_model, 'in_file')
    workflow.connect(generate_model, 'design_file', estimate_model, 'design_file')
    workflow.connect(generate_model, 'con_file', estimate_model, 'tcon_file')
    
    '''
    if 'F' in [x[1] for x in exec_get_contrasts.outputs['contrasts']]:
        outputnode = pe.MapNode(Function(input_names=['bids_dir', 'output_dir', 'contrast_pattern', 'contrasts', 'entities',
                                                  'effects', 'variances', 'z', 't', 'F', 'dof'],
                                         output_names=['effects', 'variances', 'z', 'p', 't', 'dof', 'F'],
                                         function=rename_outputs), 
                                iterfield=['entities', 'effects', 'variances', 'z', 't', 'F', 'dof'],
                                name='outputnode')
        outputnode.inputs.bids_dir = bids_dir
        outputnode.inputs.output_dir = output_dir
        outputnode.inputs.contrast_pattern = contrast_pattern
        workflow.connect(estimate_model, 'fstats', outputnode, 'F')
    '''
        
    workflow.connect(bdg, ('func', get_entities), outputnode, 'entities')
    workflow.connect(estimate_model, 'copes', outputnode, 'effects')
    workflow.connect(estimate_model, 'varcopes', outputnode, 'variances')
    workflow.connect(estimate_model, 'zstats', outputnode, 'z')
    workflow.connect(estimate_model, 'tstats', outputnode, 't')
    workflow.connect(estimate_model, 'dof_file', outputnode, 'dof')
    workflow.connect(exec_get_contrasts, 'contrasts', outputnode, 'contrasts')

        
    return workflow
"""
def fsl_second_level_wf(model,
                        step,
                        bids_dir, 
                        output_dir,
                        work_dir,
                        subject_id,
                        smoothing_fwhm=None,
                        smoothing_level=None,
                        smoothing_type=None,
                        derivatives,
                        name='fsl_second_level_wf'):
    '''Defines secondlvl workflow for emu project'''
    workflow = Workflow(name='scndlvl_wf')


    # Create a datasource node to get the task_mri and motion-noise files
    datasource = pe.Node(DataGrabber(infields=['subject_id'], outfields=info.keys()),
                         name='datasource')
    datasource.inputs.template = '*'
    datasource.inputs.subject_id = subject_id
    datasource.inputs.sort_filelist = True
    datasource.inputs.raise_on_empty = True

    # Create an Inputspec node to deal with copes and varcopes doublelist issues
    fixedfx_inputspec = pe.Node(IdentityInterface(fields=['copes', 'varcopes', 'dof_files'],
                                                  mandatory_inputs=True),
                                name="fixedfx_inputspec")
    workflow.connect(datasource, ('copes', doublelist), fixedfx_inputspec, "copes")
    workflow.connect(datasource, ('varcopes', doublelist), fixedfx_inputspec, "varcopes")
    workflow.connect(datasource, ('dof_files', doublelist), fixedfx_inputspec, "dof_files")

    # Create a Merge node to collect all of the COPES
    copemerge = pe.MapNode(Merge(), iterfield=['in_files'], name='copemerge')
    copemerge.inputs.dimension = 't'
    copemerge.inputs.environ = {'FSLOUTPUTTYPE': 'NIFTI_GZ'}
    copemerge.inputs.output_type = 'NIFTI_GZ'
    workflow.connect(fixedfx_inputspec, 'copes', copemerge, 'in_files')

    # Create a Function node to generate a DOF volume
    gendofvolume = pe.Node(Function(input_names=['dof_files', 'cope_files'],
                                    output_names=['dof_volumes'],
                                    function=get_dofvolumes),
                           name='gendofvolume')
    workflow.connect(fixedfx_inputspec, 'dof_files', gendofvolume, 'dof_files')
    workflow.connect(copemerge, 'merged_file', gendofvolume, 'cope_files')

    # Create a Merge node to collect all of the VARCOPES
    varcopemerge = pe.MapNode(Merge(), iterfield=['in_files'], name='varcopemerge')
    varcopemerge.inputs.dimension = 't'
    varcopemerge.inputs.environ = {'FSLOUTPUTTYPE': 'NIFTI_GZ'}
    varcopemerge.inputs.output_type = 'NIFTI_GZ'
    workflow.connect(fixedfx_inputspec, 'varcopes', varcopemerge, 'in_files')

    # Create a node to define the contrasts from the names of the copes
    getcontrasts = pe.Node(Function(input_names=['data_inputs'],
                                    output_names=['contrasts'],
                                    function=get_contrasts),
                           name='getcontrasts')
    workflow.connect(datasource, ('copes', doublelist), getcontrasts, 'data_inputs')

    # Create a Function node to rename output files with something more meaningful
    getsubs = pe.Node(Function(input_names=['cons'],
                               output_names=['subs'],
                               function=get_subs),
                      name='getsubs')
    workflow.connect(getcontrasts, 'contrasts', getsubs, 'cons')

    # Create a l2model node for the Fixed Effects analysis (aka within subj across runs)
    l2model = pe.MapNode(L2Model(), iterfield=['num_copes'], name='l2model')
    workflow.connect(datasource, ('copes', num_copes), l2model, 'num_copes')

    # Create a FLAMEO Node to run the fixed effects analysis
    flameo_fe = pe.MapNode(FLAMEO(),
                           iterfield=['cope_file', 'var_cope_file', 'dof_var_cope_file',
                                      'design_file', 't_con_file', 'cov_split_file'],
                           name='flameo_fe')
    flameo_fe.inputs.environ = {'FSLOUTPUTTYPE': 'NIFTI_GZ'}
    flameo_fe.inputs.log_dir = 'stats'
    flameo_fe.inputs.output_type = 'NIFTI_GZ'
    flameo_fe.inputs.run_mode = 'fe'
    workflow.connect(varcopemerge, 'merged_file', flameo_fe, 'var_cope_file')
    workflow.connect(l2model, 'design_mat', flameo_fe, 'design_file')
    workflow.connect(l2model, 'design_con', flameo_fe, 't_con_file')
    workflow.connect(l2model, 'design_grp', flameo_fe, 'cov_split_file')
    workflow.connect(gendofvolume, 'dof_volumes', flameo_fe, 'dof_var_cope_file')
    workflow.connect(datasource, 'mask_file', flameo_fe, 'mask_file')
    workflow.connect(copemerge, 'merged_file', flameo_fe, 'cope_file')

    # Create an outputspec node
    scndlvl_outputspec = Node(IdentityInterface(fields=['res4d', 'copes',
                                                        'varcopes', 'zstats',
                                                        'tstats'],
                                                mandatory_inputs=True),
                              name='scndlvl_outputspec')
    workflow.connect(flameo_fe, 'res4d', scndlvl_outputspec, 'res4d')
    workflow.connect(flameo_fe, 'copes', scndlvl_outputspec, 'copes')
    workflow.connect(flameo_fe, 'var_copes', scndlvl_outputspec, 'varcopes')
    workflow.connect(flameo_fe, 'zstats', scndlvl_outputspec, 'zstats')
    workflow.connect(flameo_fe, 'tstats', scndlvl_outputspec, 'tstats')
"""    
def clean_trial_contrasts(initial_contrasts, trial_contrasts):
    contrasts = [x for x in initial_contrasts if x in trial_contrasts or any([y in x for y in ('_gt_', '_lt_', '_vs_')])]
    return contrasts
                 
def get_contrasts(step, include_contrasts):
    """
    Produces contrasts from a given model file and a run specific events file
    """
    import itertools as it
    include_combos = list(it.combinations(include_contrasts, 2))
    all_contrasts = []
    contrasts = step["Contrasts"]
    dummy_contrasts = step["DummyContrasts"]['Conditions']
    for contrast in dummy_contrasts:
        if contrast not in include_contrasts:
            continue
        all_contrasts.append((contrast, 'T',
                              [contrast.split('.')[-1]],
                              [1]))
    #[t for x, y in include_combos for t in step['Contrasts'] if all([x in t['ConditionList'], y in t['ConditionList']])]
    for contrast in contrasts:
        if not any([all([x in contrast['ConditionList'], y in contrast['ConditionList']]) for x, y in include_combos])\
        and len(contrast['ConditionList']) == 2:
            continue
        condition_list = [x.split('.')[-1] if '.' in x else x for x in contrast['ConditionList']]
        all_contrasts.append((contrast['Name'], contrast['Type'].upper(),
                              condition_list,
                              contrast['Weights']))
    return all_contrasts

def get_info(func, confounds, events, confound_regressors):
    '''Grabs EV files for subject based on contrasts of interest'''
    from nipype.interfaces.base import Bunch
    from glob import glob
    import pandas as pd
    import os
    from copy import deepcopy
    import numpy as np
    event_data = pd.read_csv(events, sep='\t')
    conf_data = pd.read_csv(confounds, sep='\t')
    names = []
    onsets = []
    amplitudes = []
    durations = []
    regressor_names = []
    regressors = []
    for trial_type, trial_frame in event_data.groupby('trial_type'):
        if len(trial_frame) > 0:
            names.append(trial_type)
            onsets.append(trial_frame['onset'].values)
            durations.append(trial_frame['duration'].values)
            amplitudes.append(np.ones(len(trial_frame)))
    for confound in confound_regressors:
        regressor_names.append(confound)
        regressors.append(conf_data[confound].values)

    output = Bunch(conditions=names,
                   onsets=onsets,
                   durations=durations,
                   amplitudes=amplitudes,
                   tmod=None,
                   pmod=None,
                   regressor_names=regressor_names,
                   regressors=regressors)
    return output, names

def get_confounds(func):
    #A workaround to a current issue in pybids that causes massive resource use when indexing derivative tsv files
    lead = func.split('desc')[0]
    confound_file = lead + 'desc-confounds_regressors.tsv'
    return confound_file

def get_metadata(func):
    import json
    import nibabel as nb
    num_timepoints = nb.load(func).get_data().shape[3]
    lead = func.split('.nii.gz')[0]
    metafile = lead + '.json'
    with open(metafile) as mf:
        metadata = json.load(mf)
     
    return metadata['RepetitionTime'], num_timepoints

def get_motion_parameters(confounds):
    import os
    import pandas as pd
    motion_params = os.path.join(os.getcwd(), os.path.basename(confounds).split('.')[0] + '_motparams.tsv')
    confound_data = pd.read_csv(confounds, sep='\t')
    #Motion data gets formatted FSL style, with x, y, z rotation, then x,y,z translation
    motion_data = confound_data[['rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z']]
    motion_data.to_csv(motion_params, sep='\t', header=None, index=None)
    return motion_params
    
def get_smoothing_info_fsl(func, brain_mask):
    import os
    import nibabel as nb
    import numpy as np
    img = nb.load(func)
    img_data = img.get_data()
    mask_img_data = nb.load(brain_mask).get_data()
    img_affine = img.affine
    img_median = np.median(img_data[mask_img_data > 0])
    img_mean = img_data.mean(axis=3) #average on time axis
    mean_img = nb.nifti1.Nifti1Image(img_mean, img_affine)
    mean_path = os.path.join(os.getcwd(), os.path.basename(func).split('.')[0] + '_tmean.nii.gz')
    mean_img.to_filename(mean_path)
    btthresh = img_median * 0.75
    usans = [tuple([mean_path, btthresh])]
    
    return usans, btthresh

def get_entities(func):
    import os
    run_entities = []
    for ix, func_file in enumerate(func):
        stem = os.path.basename(func_file).split('.')[0]
        entity_pairs = stem.split('_')
        entities = {x.split('-')[0]:x.split('-')[1] if '-' in x else None for x in entity_pairs}
        for item in entities:
            if entities[item] == None:
                entities.pop(item)
                break
        entities['suffix'] = item
        entities['subject'] = entities.pop('sub', None)
        run_entities.append(entities)
    return run_entities

def rename_outputs(bids_dir, output_dir, contrast_pattern, contrasts, entities, 
                   effects=[], variances=[], z=[], p=[], t=[], F=[], dof=[]):
    import os
    import subprocess
    import re
    import shutil
    from bids import BIDSLayout
    def snake_to_camel(string):
        string.replace('.','_')
        words = string.replace('.','').split('_')
        return words[0] + ''.join(word.title() for word in words[1:]) 
    stat_dict = dict(effects = effects,
                     variances = variances,
                     z = z,    
                     t = t,
                     F = F)
    dof_pattern = '[sub-{subject}/][ses-{session}/][sub-{subject}_][ses-{session}_]task-{task}[_acq-{acquisition}][_rec-{reconstruction}]'\
                   '[_run-{run}][_echo-{echo}][_space-{space}]_contrast-{contrast}_dof.tsv'
    layout = BIDSLayout(bids_dir, validate=False)
    
    output_path = os.path.join(output_dir, 'run_level')
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'sub-' + entities["subject"]), exist_ok=True)
    if 'session' in entities:
        os.makedirs(os.path.join(output_path, 'sub-' + entities["subject"], 'ses-' + entities["session"]), exist_ok=True)
    outputs = {'p':[], 'dof':[]}
    contrast_names = [x[0] for x in contrasts]
    for stat, file_list in stat_dict.items():
        outputs[stat] = []
        for ix, file in enumerate(file_list):
            entities['contrast'] = snake_to_camel(contrast_names[ix])
            entities['stat'] = stat
            dest_path = os.path.join(output_path, layout.build_path(entities, contrast_pattern, validate=False))
            #os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy(file, dest_path)
            outputs[stat].append(dest_path)
            if stat == 'z':
                entities['stat'] = 'p'
                dest_path = os.path.join(output_path, layout.build_path(entities, contrast_pattern, validate=False))
                outputs['p'].append(dest_path)
                subprocess.Popen(['fslmaths', f'{file}', '-ztop', f'{dest_path}']).wait()
            #Produce dof file if one doesn't exist for a contrast
            dest_path = os.path.join(output_path, layout.build_path(entities, dof_pattern, validate=False))
            if not os.path.isfile(dest_path):
                shutil.copy(dof, dest_path)
                outputs['dof'].append(dest_path)
    effects = outputs['effects']
    variances = outputs['variances']
    z = outputs['z']
    p = outputs['p']
    t = outputs['t']
    F = outputs['F']
    dof = outputs['dof']
    return effects, variances, z, p, t, dof, F

def reshape_ra(outlier_files, confounds, confound_regressors, num_timepoints):
    import os
    import pandas as pd
    import numpy as np
    
    art_dict = {}
    outlier_frame = data = pd.read_csv(outlier_files, header=None)
    confound_frame = pd.read_csv(confounds, sep='\t')
    for i, row in outlier_frame.iterrows():
        art_dict['rapidart' + str(i)] = np.zeros(num_timepoints)
        art_dict['rapidart' + str(i)][row.values[0]] = 1
        confound_regressors.append('rapidart' + str(i))
    rapid_frame = pd.DataFrame(art_dict)
    confound_frame = pd.concat([confound_frame, rapid_frame], axis=1)
    confounds = os.path.join(os.getcwd(), os.path.basename(confounds).split('.')[0] + '_ra.tsv')
    confound_frame.to_csv(confounds, sep='\t')
    
    return confounds, confound_regressors

def snake_to_camel(string):
    string.replace('.','_')
    words = string.replace('.','').split('_')
    return words[0] + ''.join(word.title() for word in words[1:])     