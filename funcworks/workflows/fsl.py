import os
from pathlib import Path
from nipype.pipeline import engine as pe
from nipype.interfaces import fsl, io
from nipype.interfaces.utility import Function, IdentityInterface
from nipype.algorithms import modelgen, rapidart as ra
from ..interfaces.bids import (BIDSDataSink)
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

    """
    # # TODO:  Implement design matrix parser to produce figures and retrieve matrix
    #design_matrix_pattern = \
    #('[sub-{subject}/][ses-{session}/]'
    # '[sub-{subject}_][ses-{session}_]task-{task}[_acq-{acquisition}]'
    # '[_rec-{reconstruction}][_run-{run}][_echo-{echo}]_{suffix<design>}.tsv')

    bids_dir = Path(bids_dir)
    work_dir = Path(work_dir)
    workflow = pe.Workflow(name=name)

    space = None
    if 'space' in model['Input']:
        space = model['Input']['space']
    workflow.__desc__ = ""
    workflow.base_dir = work_dir / model['Name']
    confound_names = \
    [x for x in step['Model']['X'] if x not in step['DummyContrasts']['Conditions']]
    condition_names = step['DummyContrasts']['Conditions']

    bdg = pe.Node(io.BIDSDataGrabber(base_dir=bids_dir, subject=subject_id,
                                     extra_derivatives=derivatives, index_derivatives=True,
                                     output_query={'func': {'datatype':'func', 'desc':'preproc',
                                                            'extension':'nii.gz',
                                                            'suffix':'bold',
                                                            'task': model['Input']['task'],
                                                            'space':space},
                                                   'events': {'datatype':'func', 'suffix':'events',
                                                              'extension':'tsv',
                                                              'task': model['Input']['task']},
                                                   'brain_mask': {'datatype': 'func', 'desc': 'brain',
                                                                  'extension': 'nii.gz',
                                                                  'suffix':'mask',
                                                                  'task': model['Input']['task'],
                                                                  'space':space}}),
                  name='bdg')

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

    specify_model = pe.MapNode(modelgen.SpecifyModel(high_pass_filter_cutoff=-1.0,
                                                     input_units='secs'),
                               iterfield=['functional_runs', 'subject_info', 'time_repetition'],
                               name='specify_model')

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

    generate_model = pe.MapNode(fsl.FEATModel(environ={'FSLOUTPUTTYPE': 'NIFTI_GZ'},
                                              output_type='NIFTI_GZ'),
                                iterfield=['fsf_file', 'ev_files'],
                                name='generate_model')

    estimate_model = pe.MapNode(fsl.FILMGLS(environ={'FSLOUTPUTTYPE': 'NIFTI_GZ'},
                                            mask_size=5,
                                            output_type='NIFTI_GZ',
                                            results_dir='results',
                                            smooth_autocorr=True),
                                iterfield=['design_file', 'in_file', 'tcon_file'],
                                name='estimate_model')

    outputnode = pe.MapNode(Function(input_names=['output_dir',
                                                  'contrasts', 'entities',
                                                  'effects', 'variances',
                                                  'zstats', 'tstats', 'dof'],
                                     output_names=['effects', 'variances', 'zstats',
                                                   'pstats', 'tstats', 'dof', 'fstats'],
                                     function=utils.rename_outputs),
                            iterfield=['entities', 'effects', 'variances',
                                       'zstats', 'tstats', 'dof', 'contrasts'],
                            name='outputnode')
    outputnode.inputs.output_dir = output_dir

    image_pattern = ('[sub-{subject}/][ses-{ses}/]'
                     '[sub-{subject}_][ses-{ses}_]task-{task}[_acq-{acquisition}]'
                     '[_rec-{reconstruction}][_run-{run}][_echo-{echo}][_space-{space}]'
                     '_contrast-{contrast}_stat-{stat<effect|variance|z|p|t|F>}_{suffix}.nii.gz')

    ds_effects = pe.MapNode(
        BIDSDataSink(base_directory=output_dir,
                     fixed_entities={'stat':'effect', 'suffix':'statmap'},
                     path_patterns=image_pattern),
        iterfield=['entities', 'in_file'],
        run_without_submitting=True,
        name='ds_effects')
    ds_variances = pe.MapNode(
        BIDSDataSink(base_directory=output_dir,
                     fixed_entities={'stat':'variance', 'suffix':'statmap'},
                     path_patterns=image_pattern),
        iterfield=['entities', 'in_file'],
        run_without_submitting=True,
        name='ds_variances')
    ds_zstats = pe.MapNode(
        BIDSDataSink(base_directory=output_dir, fixed_entities={'stat':'z', 'suffix':'statmap'},
                     path_patterns=image_pattern),
        iterfield=['entities', 'in_file'],
        run_without_submitting=True,
        name='ds_zstats')
    ds_tstats = pe.MapNode(
        BIDSDataSink(base_directory=output_dir, fixed_entities={'stat':'t', 'suffix':'statmap'},
                     path_patterns=image_pattern),
        iterfield=['entities', 'in_file'],
        run_without_submitting=True,
        name='ds_tstats')
    ds_fstats = pe.MapNode(
        BIDSDataSink(base_directory=output_dir, fixed_entities={'stat':'F', 'suffix':'statmap'},
                     path_patterns=image_pattern),
        iterfield=['entities', 'in_file'],
        run_without_submitting=True,
        name='ds_fstats')

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

    get_tmean_img = pe.MapNode(fsl.ImageMaths(op_string='-Tmean',
                                              suffix='_mean'),
                               iterfield=['in_file'],
                               name='get_tmean_img')

    setup_susan = pe.MapNode(Function(input_names=['func', 'brain_mask', 'mean_image'],
                                      output_names=['usans', 'brightness_threshold'],
                                      function=utils.get_smoothing_info_fsl),
                             iterfield=['func', 'brain_mask', 'mean_image'],
                             name='setup_susan')

    run_susan = pe.MapNode(fsl.SUSAN(),
                           iterfield=['in_file', 'brightness_threshold', 'usans'],
                           name='run_susan')
    run_susan.inputs.fwhm = smoothing_fwhm

    apply_mask_smooth = pe.MapNode(fsl.ImageMaths(suffix='_bet',
                                                  op_string='-mas'),
                                   iterfield=['in_file', 'in_file2'],
                                   name='apply_mask_smooth')
    #Setup connections among workflow nodes

    workflow.connect([
        (bdg, apply_brainmask, [('func', 'in_file')]),
        (bdg, apply_brainmask, [('brain_mask', 'in_file2')]),
        (bdg, exec_get_metadata, [('func', 'func')]),
        (bdg, exec_get_confounds, [('func', 'func')])
    ])

    if use_rapidart:
        workflow.connect([
            (exec_get_confounds, exec_get_motion_parameters, [('confounds_file', 'confounds')]),
            (exec_get_motion_parameters, run_rapidart,
             [('motion_params', 'realignment_parameters')]),
            (bdg, run_rapidart, [('func', 'realigned_files')]),
            (bdg, run_rapidart, [('brain_mask', 'mask_file')]),
            (run_rapidart, reshape_rapidart, [('outlier_files', 'outlier_files')]),
            (exec_get_confounds, reshape_rapidart, [('confounds_file', 'confounds')]),
            (exec_get_metadata, reshape_rapidart, [('num_timepoints', 'num_timepoints')]),
            (reshape_rapidart, exec_get_info, [('confounds', 'confounds')]),
            (reshape_rapidart, exec_get_info, [('confound_regressors', 'confound_regressors')])
        ])

        exec_get_info.iterfield = ['events', 'confounds', 'confound_regressors']

    else:
        workflow.connect(exec_get_confounds, 'confounds_file', exec_get_info, 'confounds')
        exec_get_info.inputs.confound_regressors = confound_names

    exec_get_info.inputs.condition_names = condition_names
    workflow.connect(bdg, 'events', exec_get_info, 'events')

    if smoothing_level == 'l1':
        workflow.connect([
            (apply_brainmask, 'out_file', get_tmean_img, 'in_file'),
            (apply_brainmask, 'out_file', setup_susan, 'func'),
            (bdg, 'brain_mask', setup_susan, 'brain_mask'),
            (get_tmean_img, 'out_file', setup_susan, 'mean_image'),
            (apply_brainmask, 'out_file', run_susan, 'in_file'),
            (setup_susan, 'brightness_threshold', run_susan, 'brightness_threshold'),
            (setup_susan, 'usans', run_susan, 'usans'),
            (run_susan, 'smoothed_file', apply_mask_smooth, 'in_file'),
            (bdg, 'brain_mask', apply_mask_smooth, 'in_file2'),
            (apply_mask_smooth, 'out_file', specify_model, 'functional_runs'),
            (apply_mask_smooth, 'out_file', fit_model, 'functional_data')
        ])
    else:
        workflow.connect([
            (apply_brainmask, 'out_file', specify_model, 'functional_runs'),
            (apply_brainmask, 'out_file', fit_model, 'functional_data')
        ])

    workflow.connect([
        (exec_get_info, 'output', specify_model, 'subject_info'),
        (exec_get_metadata, 'repetition_time', specify_model, 'time_repetition'),
        (exec_get_info, 'names', exec_get_contrasts, 'include_contrasts'),
        (specify_model, 'session_info', fit_model, 'session_info'),
        (exec_get_metadata, 'repetition_time', fit_model, 'interscan_interval'),
        (exec_get_contrasts, 'contrasts', fit_model, 'contrasts'),
        (fit_model, 'interscan_interval', first_level_design, 'interscan_interval'),
        (fit_model, 'session_info', first_level_design, 'session_info'),
        (fit_model, 'contrasts', first_level_design, 'contrasts'),
        (fit_model, 'bases', first_level_design, 'bases'),
        (fit_model, 'model_serial_correlations', first_level_design, 'model_serial_correlations'),
        (first_level_design, 'fsf_files', generate_model, 'fsf_file'),
        (first_level_design, 'ev_files', generate_model, 'ev_files'),
        (fit_model, 'film_threshold', estimate_model, 'threshold'),
        (fit_model, 'functional_data', estimate_model, 'in_file'),
        (generate_model, 'design_file', estimate_model, 'design_file'),
        (generate_model, 'con_file', estimate_model, 'tcon_file'),
        (bdg, ('func', utils.get_entities), outputnode, 'entities'),
        (estimate_model, 'copes', ds_effects, 'in_file'),
        (estimate_model, 'varcopes', ds_variances, 'in_file'),
        (estimate_model, 'zstats', ds_zstats, 'in_file'),
        (estimate_model, 'tstats', ds_tstats, 'in_file'),
        (estimate_model, 'fstats', ds_fstats, 'in_file'),
        #(estimate_model, 'dof_file', outputnode, 'dof'),
        #(exec_get_contrasts, 'contrasts', outputnode, 'contrasts')
    ])

    return workflow


def fsl_session_level_wf(output_dir,
                         subject_id,
                         work_dir,
                         derivatives,
                         model,
                         step,
                         bids_dir,
                         name='fsl_session_level_wf'):
    """
    This workflow generates processes functional_data across a single session (read: between runs)
    and computes effects, variances, residuals and statmaps
    using FSLs FLAME0 given information in the bids model file

    """
    workflow = pe.Workflow(name=name)
    workflow.base_dir = work_dir
    workflow.desc = ""
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
