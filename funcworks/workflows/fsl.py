import os
from pathlib import Path
from nipype.pipeline import engine as pe
from nipype.interfaces import fsl, io
from nipype.interfaces.utility import Function, IdentityInterface
from nipype.algorithms import modelgen, rapidart as ra
from .. import utils
# pylint: disable=C0415,R0915,R0914,R0913
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
    contrasts = a dictionary containing contrast names as keys,
                with each value containing information for FSL Featmodel
                an example being {'contrast1_gt_contrast2' : 'T'}
                where contrast1 and contrast2 are trial_types from the events.tsv
    confounds = a list of confounds to use in regression model

    Parameters
    ----------
    omp_nthreads

    """
    # # TODO:  Implement design matrix parser to produce figures and retrieve matrix
    #design_matrix_pattern = \
    #('[sub-{subject}/][ses-{session}/]'
    # '[sub-{subject}_][ses-{session}_]task-{task}[_acq-{acquisition}]'
    # '[_rec-{reconstruction}][_run-{run}][_echo-{echo}]_{suffix<design>}.tsv')

    bids_dir = Path(bids_dir)
    work_dir = Path(work_dir)
    workflow = pe.Workflow(name=name)

    workflow.__desc__ = ""
    workflow.base_dir = work_dir / model['Name']
    confound_names = \
    [x for x in step['Model']['X'] if x not in step['DummyContrasts']['Conditions']]
    condition_names = step['DummyContrasts']['Conditions']

    bdg = pe.Node(io.BIDSDataGrabber(), name='bdg')
    bdg.inputs.base_dir = bids_dir
    bdg.inputs.subject = subject_id
    bdg.inputs.extra_derivatives = derivatives
    bdg.inputs.index_derivatives = True
    bdg.inputs.output_query = {'func': {'datatype':'func', 'desc':'preproc',
                                        'extension':'nii.gz',
                                        'suffix':'bold',
                                        'task': model['Input']['task'], 'space':None},
                               'events': {'datatype':'func', 'suffix':'events', 'extension':'tsv',
                                          'task': model['Input']['task']},
                               'brain_mask': {'datatype': 'func', 'desc': 'brain',
                                              'extension': 'nii.gz',
                                              'suffix':'mask', 'task': model['Input']['task'],
                                              'space':None}}

    exec_get_metadata = pe.MapNode(Function(input_names=['func'],
                                            output_names=['repetition_time', 'num_timepoints'],
                                            function=utils.get_metadata),
                                   iterfield=['func'],
                                   name='exec_get_metadata')

    exec_get_confounds = pe.MapNode(Function(input_names=['func'],
                                             output_names=['confounds_file'],
                                             function=utils.get_confounds),
                                    iterfield=['func'],
                                    name='exec_get_confounds')

    apply_brainmask = pe.MapNode(fsl.ImageMaths(suffix='_bet',
                                                op_string='-mas'),
                                 iterfield=['in_file', 'in_file2'],
                                 name='apply_brainmask')

    exec_get_info = pe.MapNode(Function(input_names=['events',
                                                     'confounds',
                                                     'confound_regressors',
                                                     'condition_names'],
                                        output_names=['output', 'names'],
                                        function=utils.get_info),
                               iterfield=['events', 'confounds'],
                               name='exec_get_info')

    exec_get_contrasts = pe.MapNode(Function(input_names=['step', 'include_contrasts'],
                                             output_names=['contrasts'],
                                             function=utils.get_contrasts),
                                    iterfield=['include_contrasts'],
                                    name='exec_get_contrasts')
    exec_get_contrasts.inputs.step = step

    specify_model = pe.MapNode(modelgen.SpecifyModel(),
                               iterfield=['functional_runs', 'subject_info', 'time_repetition'],
                               name='specify_model')
    specify_model.inputs.high_pass_filter_cutoff = -1.0
    specify_model.inputs.input_units = 'secs'

    fit_model = pe.MapNode(IdentityInterface(fields=['session_info', 'interscan_interval',
                                                     'contrasts', 'film_threshold',
                                                     'functional_data', 'bases',
                                                     'model_serial_correlations'],
                                             mandatory_inputs=True),
                           iterfield=['functional_data', 'session_info',
                                      'interscan_interval', 'contrasts'],
                           name='fit_model')
    fit_model.inputs.bases = {'dgamma':{'derivs': False}}
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

    outputnode = pe.MapNode(Function(input_names=['bids_dir', 'output_dir',
                                                  'contrasts', 'entities',
                                                  'effects', 'variances', 'zstats', 'tstats', 'dof'],
                                     output_names=['effects', 'variances', 'zstats',
                                                   'pstats', 'tstats', 'dof', 'fstats'],
                                     function=utils.rename_outputs),
                            iterfield=['entities', 'effects', 'variances',
                                       'zstats', 'tstats', 'dof', 'contrasts'],
                            name='outputnode')
    outputnode.inputs.bids_dir = bids_dir
    outputnode.inputs.output_dir = output_dir

    #Setup connections among workflow nodes
    workflow.connect(bdg, 'func', apply_brainmask, 'in_file')
    workflow.connect(bdg, 'brain_mask', apply_brainmask, 'in_file2')

    workflow.connect(bdg, 'func', exec_get_metadata, 'func')

    workflow.connect(bdg, 'func', exec_get_confounds, 'func')

    if use_rapidart:
        exec_get_motion_parameters = pe.MapNode(Function(input_names=['confounds'],
                                                         output_names='motion_params',
                                                         function=utils.get_motion_parameters),
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

        reshape_rapidart = pe.MapNode(Function(input_names=['outlier_files', 'confounds',
                                                            'confound_regressors',
                                                            'num_timepoints'],
                                               output_names=['confounds', 'confound_regressors'],
                                               function=utils.reshape_ra),
                                      iterfield=['outlier_files', 'confounds', 'num_timepoints'],
                                      name='reshape_rapidart')
        reshape_rapidart.inputs.confound_regressors = confound_names

        workflow.connect(exec_get_confounds, 'confounds_file',
                         exec_get_motion_parameters, 'confounds')

        workflow.connect(exec_get_motion_parameters, 'motion_params',
                         run_rapidart, 'realignment_parameters')
        workflow.connect(bdg, 'func', run_rapidart, 'realigned_files')
        workflow.connect(bdg, 'brain_mask', run_rapidart, 'mask_file')

        workflow.connect(run_rapidart, 'outlier_files', reshape_rapidart, 'outlier_files')
        workflow.connect(exec_get_confounds, 'confounds_file', reshape_rapidart, 'confounds')
        workflow.connect(exec_get_metadata, 'num_timepoints', reshape_rapidart, 'num_timepoints')

        exec_get_info.iterfield = ['events', 'confounds', 'confound_regressors']

        workflow.connect(reshape_rapidart, 'confounds', exec_get_info, 'confounds')
        workflow.connect(reshape_rapidart, 'confound_regressors',
                         exec_get_info, 'confound_regressors')

    else:
        workflow.connect(exec_get_confounds, 'confounds_file', exec_get_info, 'confounds')
        exec_get_info.inputs.confound_regressors = confound_names

    exec_get_info.inputs.condition_names = condition_names
    workflow.connect(bdg, 'events', exec_get_info, 'events')

    if smoothing_level == 'l1':
        get_tmean_img = pe.MapNode(fsl.ImageMaths(op_string='-Tmean',
                                                  suffix='_mean'),
                                   iterfield=['in_file'],
                                   name='smooth_meanfunc')

        setup_susan = pe.MapNode(Function(input_names=['func', 'brain_mask', 'mean_image'],
                                          output_names=['usans', 'brightness_threshold'],
                                          function=utils.get_smoothing_info_fsl),
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

        workflow.connect(apply_brainmask, 'out_file', get_tmean_img, 'in_file')

        workflow.connect(apply_brainmask, 'out_file', setup_susan, 'func')
        workflow.connect(bdg, 'brain_mask', setup_susan, 'brain_mask')
        workflow.connect(get_tmean_img, 'out_file', setup_susan, 'mean_img')

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
        outputnode = pe.MapNode(Function(input_names=['bids_dir', 'output_dir',
                                                      'contrasts', 'entities',
                                                      'effects', 'variances', 'zstats', 'tstats', 'stats', 'dof'],
                                         output_names=['effects', 'variances', 'zstats',
                                                       'pstats', 'tstats', 'dof', 'fstats'],
                                         function=utils.rename_outputs),
                                iterfield=['entities', 'effects', 'variances',
                                           'zstats', 'tstats', 'stats', 'dof'],
                                name='outputnode')
        outputnode.inputs.bids_dir = bids_dir
        outputnode.inputs.output_dir = output_dir
        workflow.connect(estimate_model, 'fstats', outputnode, 'fstats')
    '''

    workflow.connect(bdg, ('func', utils.get_entities), outputnode, 'entities')
    workflow.connect(estimate_model, 'copes', outputnode, 'effects')
    workflow.connect(estimate_model, 'varcopes', outputnode, 'variances')
    workflow.connect(estimate_model, 'zstats', outputnode, 'zstats')
    workflow.connect(estimate_model, 'tstats', outputnode, 'tstats')
    workflow.connect(estimate_model, 'dof_file', outputnode, 'dof')
    workflow.connect(exec_get_contrasts, 'contrasts', outputnode, 'contrasts')


    return workflow


def fsl_session_level_wf(output_dir,
                         subject_id,
                         work_dir,
                         derivatives,
                         name='fsl_session_level_wf'):

    workflow = pe.Workflow(name=name)
    workflow.base_dir = work_dir

    return_conts = pe.Node(Function(input_names=['subject_id', 'derivatives'],
                                    output_names=['effects', 'variances', 'dofs'],
                                    function=utils.return_contrasts), name='return_conts')
    return_conts.inputs.derivatives = derivatives
    return_conts.inputs.subject_id = subject_id

    merge_conts = pe.MapNode(Function(input_names=['effects', 'variances', 'dofs', 'derivatives'],
                                      output_names=['merged_effects',
                                                    'merged_variances',
                                                    'merged_dofs'],
                                      function=utils.merge_runs),
                             iterfield=['effects', 'variances', 'dofs'],
                             name='merge_conts')
    merge_conts.inputs.derivatives = derivatives
    workflow.connect(return_conts, 'effects', merge_conts, 'effects')
    workflow.connect(return_conts, 'variances', merge_conts, 'variances')
    workflow.connect(return_conts, 'dofs', merge_conts, 'dofs')

    count_runs = pe.MapNode(Function(input_names=['effects'],
                                     output_names=['num_copes'],
                                     function=utils.num_copes),
                            iterfield=['effects'],
                            name='count_runs')
    workflow.connect(merge_conts, 'merged_effects', count_runs, 'effects')

    model_session = pe.MapNode(fsl.L2Model(), iterfield=['num_copes'], name='model_session')
    workflow.connect(count_runs, 'num_copes', model_session, 'num_copes')

    find_brainmask = pe.Node(Function(input_names=['subject_id', 'derivatives'],
                                      output_names='brain_mask',
                                      function=utils.get_brainmask),
                             name='find_brainmask')
    find_brainmask.inputs.derivatives = derivatives
    find_brainmask.inputs.subject_id = subject_id

    # Create a FLAMEO Node to run the fixed effects analysis
    flameo_fe = pe.MapNode(fsl.FLAMEO(),
                           iterfield=['cope_file', 'var_cope_file', 'dof_var_cope_file',
                                      'design_file', 't_con_file', 'cov_split_file'],
                           name='flameo_fe')
    flameo_fe.inputs.environ = {'FSLOUTPUTTYPE': 'NIFTI_GZ'}
    flameo_fe.inputs.log_dir = 'stats'
    flameo_fe.inputs.output_type = 'NIFTI_GZ'
    flameo_fe.inputs.run_mode = 'fe'
    workflow.connect(model_session, 'design_mat', flameo_fe, 'design_file')
    workflow.connect(model_session, 'design_con', flameo_fe, 't_con_file')
    workflow.connect(model_session, 'design_grp', flameo_fe, 'cov_split_file')
    workflow.connect(merge_conts, 'merged_dofs', flameo_fe, 'dof_var_cope_file')
    workflow.connect(merge_conts, 'merged_variances', flameo_fe, 'var_cope_file')
    workflow.connect(merge_conts, 'merged_effects', flameo_fe, 'cope_file')
    workflow.connect(find_brainmask, 'brain_mask', flameo_fe, 'mask_file')

    get_renames = pe.MapNode(Function(input_names=['merged_effects', 'effects',
                                                   'variances', 'tstats',
                                                   'zstats', 'res4d'],
                                      output_names=['new_names'],
                                      function=utils.rename_contrasts),
                             iterfield=['merged_effects', 'effects',
                                        'variances', 'tstats',
                                        'zstats', 'res4d'],
                             name='get_renames')
    workflow.connect(merge_conts, 'merged_effects', get_renames, 'merged_effects')
    workflow.connect(flameo_fe, 'copes', get_renames, 'effects')
    workflow.connect(flameo_fe, 'var_copes', get_renames, 'variances')
    workflow.connect(flameo_fe, 'tstats', get_renames, 'tstats')
    workflow.connect(flameo_fe, 'zstats', get_renames, 'zstats')
    workflow.connect(flameo_fe, 'res4d', get_renames, 'res4d')

    # Create a datasink node
    sinkd = pe.MapNode(io.DataSink(infields=['copes', 'var_copes', 'tstats',
                                             'zstats', 'res4d', 'substitutions']),
                       iterfield=['copes', 'var_copes', 'tstats',
                                  'zstats', 'res4d', 'substitutions'],
                       name='sinkd')
    sinkd.inputs.base_directory = os.path.join(output_dir, 'session_level')
    sinkd.inputs.container = 'sub-' + subject_id
    workflow.connect(flameo_fe, 'copes', sinkd, 'copes')
    workflow.connect(flameo_fe, 'var_copes', sinkd, 'var_copes')
    workflow.connect(flameo_fe, 'tstats', sinkd, '.tstats')
    workflow.connect(flameo_fe, 'zstats', sinkd, '.zstats')
    workflow.connect(flameo_fe, 'res4d', sinkd, '.res4d')
    workflow.connect(get_renames, 'new_names', sinkd, 'substitutions')

    return workflow
