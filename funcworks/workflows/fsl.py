"""
Run and Session Level WFs in FSL
"""
#pylint: disable=R0913,R0914
from pathlib import Path
from nipype.pipeline import engine as pe
from nipype.interfaces import fsl, io
from nipype.interfaces.utility import Function, IdentityInterface
from nipype.algorithms import modelgen, rapidart as ra
from ..interfaces.bids import (BIDSDataSink)
from ..interfaces.io import GetModelInfo
from ..interfaces.visualization import PlotMatrices
from .. import utils

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
                       detrend_poly=None,
                       name='fsl_first_level_wf'):
    """
    This workflow generates processes function data with information given in
    the model file

    """

    bids_dir = Path(bids_dir)
    work_dir = Path(work_dir)
    workflow = pe.Workflow(name=name)

    if smoothing_type == 'iso':
        dimensionality = 3
    elif smoothing_type == 'inp':
        dimensionality = 2

    fixed_entities = model['Input']['Include']
    if 'space' not in fixed_entities:
        fixed_entities['space'] = None
    event_entities = fixed_entities.copy()
    event_entities.pop('space', None)

    workflow.__desc__ = ""
    (work_dir / model['Name']).mkdir(exist_ok=True)

    bdg = pe.Node(
        io.BIDSDataGrabber(base_dir=bids_dir, subject=subject_id,
                           extra_derivatives=derivatives, index_derivatives=True,
                           output_query={'func': {**{'datatype':'func', 'desc':'preproc',
                                                     'extension':'nii.gz', 'suffix':'bold'},
                                                  **fixed_entities},
                                         'events': {**{'datatype':'func', 'suffix':'events',
                                                       'extension':'tsv'},
                                                    **event_entities},
                                         'brain_mask': {**{'datatype': 'func', 'desc': 'brain',
                                                           'extension': 'nii.gz', 'suffix':'mask'},
                                                        **fixed_entities}}),
        name='bdg')

    get_info = pe.MapNode(
        GetModelInfo(model=step, detrend_poly=detrend_poly),
        iterfield=['functional_file', 'events_file'],
        name='get_info')

    apply_brainmask = pe.MapNode(
        fsl.ImageMaths(suffix='_bet', op_string='-mas'),
        iterfield=['in_file', 'in_file2'],
        name='apply_brainmask')

    specify_model = pe.MapNode(
        modelgen.SpecifyModel(high_pass_filter_cutoff=-1.0, input_units='secs'),
        iterfield=['functional_runs', 'subject_info', 'time_repetition'],
        name='specify_model')

    fit_model = pe.MapNode(
        IdentityInterface(fields=['session_info', 'interscan_interval',
                                  'contrasts', 'functional_data'],
                          mandatory_inputs=True),
        iterfield=['functional_data', 'session_info',
                   'interscan_interval', 'contrasts'],
        name='fit_model')

    first_level_design = pe.MapNode(
        fsl.Level1Design(bases={'dgamma':{'derivs': False}},
                         model_serial_correlations=True),
        iterfield=['session_info', 'interscan_interval', 'contrasts'],
        name='first_level_design')

    generate_model = pe.MapNode(
        fsl.FEATModel(environ={'FSLOUTPUTTYPE': 'NIFTI_GZ'},
                      output_type='NIFTI_GZ'),
        iterfield=['fsf_file', 'ev_files'],
        name='generate_model')

    estimate_model = pe.MapNode(
        fsl.FILMGLS(environ={'FSLOUTPUTTYPE': 'NIFTI_GZ'}, mask_size=5, threshold=0.0,
                    output_type='NIFTI_GZ', results_dir='results', smooth_autocorr=True),
        iterfield=['design_file', 'in_file', 'tcon_file'],
        name='estimate_model')

    image_pattern = ('[sub-{subject}/][ses-{session}/]'
                     '[sub-{subject}_][ses-{session}_]task-{task}[_acq-{acquisition}]'
                     '[_rec-{reconstruction}][_run-{run:02d}][_echo-{echo}][_space-{space}]'
                     '_contrast-{contrast}_stat-{stat<effect|variance|z|p|t|F>}_{suffix}.nii.gz')

    produce_contrast_entities = pe.MapNode(
        Function(input_names=['func_file', 'contrasts'], output_names='contrast_entities',
                 function=utils.get_entities),
        iterfield=['func_file', 'contrasts'],
        name='produce_contrast_entities')

    plot_matrices = pe.MapNode(
        PlotMatrices(output_dir=output_dir),
        iterfield=['mat_file', 'con_file', 'entities', 'run_info'],
        name='plot_matrices')

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

    run_rapidart = pe.MapNode(
        ra.ArtifactDetect(use_differences=[True, False], use_norm=True,
                          zintensity_threshold=3, norm_threshold=1,
                          bound_by_brainmask=True, mask_type='file',
                          parameter_source='FSL'),
        iterfield=['realignment_parameters', 'realigned_files', 'mask_file'],
        name='run_rapidart')

    reshape_rapidart = pe.MapNode(
        Function(input_names=['run_info', 'metadata', 'outlier_files'],
                 output_names=['run_info'],
                 function=utils.reshape_ra),
        iterfield=['outlier_files', 'run_info', 'metadata'],
        name='reshape_rapidart')

    get_tmean_img = pe.MapNode(
        fsl.ImageMaths(op_string='-Tmean', suffix='_mean'),
        iterfield=['in_file'],
        name='get_tmean_img')

    setup_susan = pe.MapNode(
        Function(input_names=['func', 'brain_mask', 'mean_image'],
                 output_names=['usans', 'brightness_threshold'],
                 function=utils.get_smoothing_info_fsl),
        iterfield=['func', 'brain_mask', 'mean_image'],
        name='setup_susan')

    run_susan = pe.MapNode(
        fsl.SUSAN(),
        iterfield=['in_file', 'brightness_threshold', 'usans'],
        name='run_susan')

    apply_mask_smooth = pe.MapNode(
        fsl.ImageMaths(suffix='_bet', op_string='-mas'),
        iterfield=['in_file', 'in_file2'],
        name='apply_mask_smooth')

    #Setup connections among nodes
    workflow.connect([
        (bdg, apply_brainmask, [('func', 'in_file')]),
        (bdg, apply_brainmask, [('brain_mask', 'in_file2')]),
        (bdg, get_info, [('func', 'functional_file')]),
        (bdg, get_info, [('events', 'events_file')])
    ])

    if use_rapidart:
        workflow.connect([
            (get_info, run_rapidart,
             [('motion_parameters', 'realignment_parameters')]),
            (bdg, run_rapidart, [('func', 'realigned_files')]),
            (bdg, run_rapidart, [('brain_mask', 'mask_file')]),
            (run_rapidart, reshape_rapidart, [('outlier_files', 'outlier_files')]),
            (get_info, reshape_rapidart, [('run_info', 'run_info')]),
            (get_info, reshape_rapidart, [('run_metadata', 'metadata')]),
            (reshape_rapidart, specify_model, [('run_info', 'subject_info')]),
            (reshape_rapidart, plot_matrices, [('run_info', 'run_info')])
        ])
    else:
        workflow.connect([
            (get_info, specify_model, [('run_info', 'subject_info')]),
            (get_info, plot_matrices, [('run_info', 'run_info')])
        ])

    if smoothing_level == 'l1':
        run_susan.inputs.fwhm = smoothing_fwhm
        run_susan.inputs.dimension = dimensionality
        workflow.connect([
            (apply_brainmask, get_tmean_img, [('out_file', 'in_file')]),
            (apply_brainmask, setup_susan, [('out_file', 'func')]),
            (bdg, setup_susan, [('brain_mask', 'brain_mask')]),
            (get_tmean_img, setup_susan, [('out_file', 'mean_image')]),
            (apply_brainmask, run_susan, [('out_file', 'in_file')]),
            (setup_susan, run_susan, [('brightness_threshold', 'brightness_threshold')]),
            (setup_susan, run_susan, [('usans', 'usans')]),
            (run_susan, apply_mask_smooth, [('smoothed_file', 'in_file')]),
            (bdg, apply_mask_smooth, [('brain_mask', 'in_file2')]),
            (apply_mask_smooth, specify_model, [('out_file', 'functional_runs')]),
            (apply_mask_smooth, fit_model, [('out_file', 'functional_data')])
        ])
    else:
        workflow.connect([
            (apply_brainmask, specify_model, [('out_file', 'functional_runs')]),
            (apply_brainmask, fit_model, [('out_file', 'functional_data')])
        ])

    workflow.connect([
        (get_info, specify_model, [('repetition_time', 'time_repetition')]),

        (specify_model, fit_model, [('session_info', 'session_info')]),
        (get_info, fit_model, [('repetition_time', 'interscan_interval')]),
        (get_info, fit_model, [('run_contrasts', 'contrasts')]),

        (fit_model, first_level_design, [('interscan_interval', 'interscan_interval')]),
        (fit_model, first_level_design, [('session_info', 'session_info')]),
        (fit_model, first_level_design, [('contrasts', 'contrasts')]),

        (first_level_design, generate_model, [('fsf_files', 'fsf_file')]),
        (first_level_design, generate_model, [('ev_files', 'ev_files')]),

        (get_info, plot_matrices, [('run_entities', 'entities')]),
        (generate_model, plot_matrices, [('design_file', 'mat_file')]),
        (generate_model, plot_matrices, [('con_file', 'con_file')]),

        (fit_model, estimate_model, [('functional_data', 'in_file')]),
        (generate_model, estimate_model, [('design_file', 'design_file')]),
        (generate_model, estimate_model, [('con_file', 'tcon_file')]),

        (bdg, produce_contrast_entities, [('func', 'func_file')]),
        (get_info, produce_contrast_entities, [('run_contrasts', 'contrasts')]),

        (produce_contrast_entities, ds_effects, [('contrast_entities', 'entities')]),
        (produce_contrast_entities, ds_variances, [('contrast_entities', 'entities')]),
        (produce_contrast_entities, ds_zstats, [('contrast_entities', 'entities')]),
        (produce_contrast_entities, ds_tstats, [('contrast_entities', 'entities')]),
        (produce_contrast_entities, ds_fstats, [('contrast_entities', 'entities')]),
        (estimate_model, ds_effects, [('copes', 'in_file')]),
        (estimate_model, ds_variances, [('varcopes', 'in_file')]),
        (estimate_model, ds_zstats, [('zstats', 'in_file')]),
        (estimate_model, ds_tstats, [('tstats', 'in_file')]),
        (estimate_model, ds_fstats, [('fstats', 'in_file')])
    ])

    return workflow

'''
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
'''
