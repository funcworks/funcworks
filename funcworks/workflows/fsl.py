"""
Run and Session Level WFs in FSL
"""
#pylint: disable=R0913,R0914
from pathlib import Path
from nipype.pipeline import engine as pe
from nipype.interfaces import fsl
from nipype.interfaces.utility import Function, IdentityInterface
from nipype.algorithms import modelgen, rapidart as ra
from ..interfaces.bids import BIDSDataGrabber, BIDSDataSink
from ..interfaces.modelgen import GetRunModelInfo, GenerateHigherInfo
from ..interfaces.io import MergeAll, CollateWithMetadata
from ..interfaces.visualization import PlotMatrices
from .. import utils

def fsl_run_level_wf(model,
                     step,
                     bids_dir,
                     output_dir,
                     work_dir,
                     subject_id,
                     derivatives,
                     database_path,
                     smoothing_fwhm=None,
                     smoothing_level=None,
                     smoothing_type=None,
                     use_rapidart=False,
                     detrend_poly=None,
                     align_volumes=None,
                     name='fsl_run_level_wf'):
    """
    This workflow generates processes function data with information given in
    the model file

    """
    bids_dir = Path(bids_dir)
    work_dir = Path(work_dir)
    workflow = pe.Workflow(name=name)

    level = step['Level']
    if smoothing_type == 'iso':
        dimensionality = 3
    elif smoothing_type == 'inp':
        dimensionality = 2

    fixed_entities = model['Input']['Include']
    if 'space' not in fixed_entities:
        fixed_entities['space'] = None
    event_entities = fixed_entities.copy()
    event_entities.pop('space', None)
    reference_entities = fixed_entities.copy()
    if align_volumes:
        reference_entities['run'] = align_volumes
    workflow.__desc__ = ""
    (work_dir / model['Name']).mkdir(exist_ok=True)

    bdg = pe.Node(
        BIDSDataGrabber(
            base_dir=bids_dir, subject=subject_id,
            index_derivatives=derivatives,
            database_path=database_path,
            output_query={'func': {**{'datatype':'func', 'desc':'preproc',
                                      'extension':'nii.gz', 'suffix':'bold'},
                                   **fixed_entities},
                          'events': {**{'datatype':'func', 'suffix':'events',
                                        'extension':'tsv'},
                                     **event_entities},
                          'brain_mask': {**{'datatype': 'func', 'desc': 'brain',
                                            'extension': 'nii.gz', 'suffix':'mask'},
                                         **reference_entities},
                          'bold_ref': {**{'datatype': 'func',
                                          'extension': 'nii.gz', 'suffix':'boldref'},
                                       **reference_entities}}),
        name='bdg')

    get_info = pe.MapNode(
        GetRunModelInfo(model=step, detrend_poly=detrend_poly),
        iterfield=['functional_file', 'events_file'],
        name='get_run_info')

    reference_outputs = pe.Node(
        Function(input_names=['brain_mask', 'bold_ref'],
                 output_names=['brain_mask', 'bold_ref'],
                 function=utils.reference_outputs),
        name='reference_outputs')

    #Realign functional runs to the first functional run
    if align_volumes:
        realign_fields = ['in_file']
        apply_bmsk = ['in_file']
        susan_fields = ['func', 'mean_image']
        apply_bmsk_smooth = ['in_file']
    else:
        realign_fields = ['in_file', 'bold_ref']
        apply_bmsk = ['in_file', 'in_file2']
        susan_fields = ['func', 'mean_image', 'brain_mask']
        apply_bmsk_smooth = ['in_file', 'in_file2']

    realign_runs = pe.MapNode(
        fsl.MCFLIRT(interpolation='sinc'),
        iterfield=realign_fields,
        name='realign_runs')

    apply_brainmask = pe.MapNode(
        fsl.ImageMaths(suffix='_bet', op_string='-mas'),
        iterfield=apply_bmsk,
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
                         model_serial_correlations=False),
        iterfield=['session_info', 'interscan_interval', 'contrasts'],
        name='first_level_design')

    generate_model = pe.MapNode(
        fsl.FEATModel(environ={'FSLOUTPUTTYPE': 'NIFTI_GZ'},
                      output_type='NIFTI_GZ'),
        iterfield=['fsf_file', 'ev_files'],
        name='generate_model')

    estimate_model = pe.MapNode(
        fsl.FILMGLS(environ={'FSLOUTPUTTYPE': 'NIFTI_GZ'}, mask_size=5, threshold=0.0,
                    output_type='NIFTI_GZ', results_dir='results', #smooth_autocorr=True
                    autocorr_noestimate=True),
        iterfield=['design_file', 'in_file', 'tcon_file'],
        name='estimate_model')

    image_pattern = ('[sub-{subject}/][ses-{session}/]'
                     '[sub-{subject}_][ses-{session}_]task-{task}[_acq-{acquisition}]'
                     '[_rec-{reconstruction}][_run-{run}][_echo-{echo}][_space-{space}]'
                     '_contrast-{contrast}_stat-{stat<effect|variance|z|p|t|F>}_statmap.nii.gz')

    run_rapidart = pe.MapNode(
        ra.ArtifactDetect(use_differences=[True, False], use_norm=True,
                          zintensity_threshold=3, norm_threshold=1,
                          bound_by_brainmask=True, mask_type='file',
                          parameter_source='FSL'),
        iterfield=['realignment_parameters', 'realigned_files'],
        name='run_rapidart')

    reshape_rapidart = pe.MapNode(
        Function(input_names=['run_info', 'func', 'outlier_files',
                              'contrast_entities'],
                 output_names=['run_info', 'contrast_entities'],
                 function=utils.reshape_ra),
        iterfield=['outlier_files', 'run_info', 'func', 'contrast_entities'],
        name='reshape_rapidart')

    get_tmean_img = pe.MapNode(
        fsl.ImageMaths(op_string='-Tmean', suffix='_mean'),
        iterfield=['in_file'],
        name='get_tmean_img')

    setup_susan = pe.MapNode(
        Function(input_names=['func', 'brain_mask', 'mean_image'],
                 output_names=['usans', 'brightness_threshold'],
                 function=utils.get_smoothing_info_fsl),
        iterfield=susan_fields,
        name='setup_susan')

    run_susan = pe.MapNode(
        fsl.SUSAN(fwhm=smoothing_fwhm, dimension=dimensionality),
        iterfield=['in_file', 'brightness_threshold', 'usans'],
        name='run_susan')

    apply_mask_smooth = pe.MapNode(
        fsl.ImageMaths(suffix='_bet', op_string='-mas'),
        iterfield=apply_bmsk_smooth,
        name='apply_mask_smooth')

    #Exists solely to correct undesirable behavior of FSL
    #that results in loss of constant columns
    correct_matrices = pe.MapNode(
        Function(input_names=['design_matrix'],
                 output_names=['design_matrix'],
                 function=utils.correct_matrix),
        iterfield=['design_matrix'],
        name='correct_matrices')

    collate = pe.Node(
        MergeAll(['effect_maps', 'variance_maps', 'tstat_maps', 'zscore_maps', 'contrast_metadata'],
                 check_lengths=True),
        name='collate_run_level')

    collate_outputs = pe.Node(
        CollateWithMetadata(
            fields=['effect_maps', 'variance_maps', 'tstat_maps', 'zscore_maps'],
            field_to_metadata_map={
                'effect_maps': {'stat': 'effect'},
                'variance_maps': {'stat': 'variance'},
                #'pvalue_maps': {'stat': 'p'},
                'zscore_maps': {'stat': 'z'},
                'tstat_maps': {'stat' : 't'}
            }),
        name=f'collate_run_outputs')

    plot_matrices = pe.MapNode(
        PlotMatrices(output_dir=output_dir),
        iterfield=['mat_file', 'con_file', 'entities', 'run_info'],
        name='plot_matrices')

    ds_contrast_maps = pe.Node(
        BIDSDataSink(base_directory=output_dir,
                     path_patterns=image_pattern),
        iterfield=['entities', 'in_file'],
        run_without_submitting=True,
        name=f'ds_{level}_contrast_maps')

    wrangle_outputs = pe.Node(
        IdentityInterface(fields=['contrast_metadata', 'contrast_maps', 'brain_mask']),
        name=f'wrangle_{level}_outputs')
    #Setup connections among nodes

    workflow.connect([
        (bdg, realign_runs, [('func', 'in_file')]),
        (bdg, reference_outputs, [('bold_ref', 'bold_ref'),
                                  ('brain_mask', 'brain_mask')]),
        (reference_outputs, realign_runs, [('bold_ref', 'ref_file')]),
        (realign_runs, apply_brainmask, [('out_file', 'in_file')]),
        (reference_outputs, apply_brainmask, [('brain_mask', 'in_file2')]),
        (bdg, get_info, [('func', 'functional_file'), ('events', 'events_file')])
    ])

    if use_rapidart:
        workflow.connect([
            (get_info, run_rapidart, [('motion_parameters', 'realignment_parameters')]),
            (reference_outputs, run_rapidart, [('brain_mask', 'mask_file')]),
            (realign_runs, run_rapidart, [('out_file', 'realigned_files')]),
            (run_rapidart, reshape_rapidart, [('outlier_files', 'outlier_files')]),
            (get_info, reshape_rapidart, [('run_info', 'run_info')]),
            (realign_runs, reshape_rapidart, [('out_file', 'func')]),
            (get_info, reshape_rapidart, [('contrast_entities', 'contrast_entities')]),
            (reshape_rapidart, specify_model, [('run_info', 'subject_info')]),
            (reshape_rapidart, plot_matrices, [('run_info', 'run_info')]),
            (reshape_rapidart, collate, [
                ('contrast_entities', 'contrast_metadata')])
        ])
    else:
        workflow.connect([
            (get_info, specify_model, [('run_info', 'subject_info')]),
            (get_info, plot_matrices, [('run_info', 'run_info')]),
            (get_info, collate, [
                ('contrast_entities', 'contrast_metadata')])
        ])

    if smoothing_level == 'l1':
        workflow.connect([
            (apply_brainmask, get_tmean_img, [('out_file', 'in_file')]),
            (apply_brainmask, setup_susan, [('out_file', 'func')]),
            (reference_outputs, setup_susan, [('brain_mask', 'brain_mask')]),
            (get_tmean_img, setup_susan, [('out_file', 'mean_image')]),
            (apply_brainmask, run_susan, [('out_file', 'in_file')]),
            (setup_susan, run_susan, [('brightness_threshold', 'brightness_threshold'),
                                      ('usans', 'usans')]),
            (run_susan, apply_mask_smooth, [('smoothed_file', 'in_file')]),
            (reference_outputs, apply_mask_smooth, [('brain_mask', 'in_file2')]),
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
        (get_info, fit_model, [('repetition_time', 'interscan_interval'),
                               ('run_contrasts', 'contrasts')]),

        (fit_model, first_level_design, [('interscan_interval', 'interscan_interval'),
                                         ('session_info', 'session_info'),
                                         ('contrasts', 'contrasts')]),

        (first_level_design, generate_model, [('fsf_files', 'fsf_file')]),
        (first_level_design, generate_model, [('ev_files', 'ev_files')]),
    ])
    if detrend_poly:
        workflow.connect([
            (generate_model, correct_matrices, [('design_file', 'design_matrix')]),
            (correct_matrices, plot_matrices, [('design_matrix', 'mat_file')]),
            (correct_matrices, estimate_model, [('design_matrix', 'design_file')])
        ])
    else:
        workflow.connect([
            (generate_model, plot_matrices, [('design_file', 'mat_file')]),
            (generate_model, estimate_model, [('design_file', 'design_file')]),
        ])

    workflow.connect([
        (get_info, plot_matrices, [('run_entities', 'entities')]),
        (generate_model, plot_matrices, [('con_file', 'con_file')]),

        (fit_model, estimate_model, [('functional_data', 'in_file')]),
        (generate_model, estimate_model, [('con_file', 'tcon_file')]),

        (estimate_model, collate, [('copes', 'effect_maps'),
                                   ('varcopes', 'variance_maps'),
                                   ('tstats', 'tstat_maps'),
                                   ('zstats', 'zscore_maps')]),

        (collate, collate_outputs, [('effect_maps', 'effect_maps'),
                                    ('variance_maps', 'variance_maps'),
                                    ('tstat_maps', 'tstat_maps'),
                                    ('zscore_maps', 'zscore_maps'),
                                    ('contrast_metadata', 'metadata')]),

        (collate_outputs, ds_contrast_maps, [('out', 'in_file'),
                                             ('metadata', 'entities')]),

        (collate_outputs, wrangle_outputs, [('metadata', 'contrast_metadata'),
                                            ('out', 'contrast_maps')]),
        (reference_outputs, wrangle_outputs, [('brain_mask', 'brain_mask')])
    ])

    return workflow


def fsl_higher_level_wf(output_dir,
                        work_dir,
                        step,
                        smoothing_fwhm=None,
                        smoothing_level=None,
                        smoothing_type=None,
                        name='fsl_higher_level_wf'):
    """
    This workflow generates processes functional_data across a single session (read: between runs)
    and computes effects, variances, residuals and statmaps
    using FSLs FLAME0 given information in the bids model file

    """

    workflow = pe.Workflow(name=name)
    workflow.base_dir = work_dir
    workflow.desc = ""

    level = step['Level']

    image_pattern = ('[sub-{subject}/][ses-{session}/]'
                     '[sub-{subject}_][ses-{session}_]task-{task}[_acq-{acquisition}]'
                     '[_rec-{reconstruction}][_echo-{echo}][_space-{space}]'
                     '_contrast-{contrast}_stat-{stat<effect|variance|z|p|t|F>}_statmap.nii.gz')

    wrangle_inputs = pe.Node(
        IdentityInterface(fields=['contrast_metadata', 'contrast_maps', 'brain_mask']),
        name=f'wrangle_{level}_inputs')

    get_info = pe.Node(
        GenerateHigherInfo(model=step),
        name=f'get_{level}_info')
    if smoothing_level == 'l2':
        pass

    estimate_model = pe.MapNode(
        fsl.FLAMEO(run_mode='fe'),
        iterfield=['design_file', 't_con_file', 'cov_split_file',
                   'dof_var_cope_file', 'var_cope_file', 'cope_file'],
        name=f'estimate_{level}_model')

    collate = pe.Node(
        MergeAll(['effect_maps', 'variance_maps', 'tstat_maps',
                  'zscore_maps', 'contrast_metadata'],
                 check_lengths=False),
        name=f'collate_{level}_level')

    collate_outputs = pe.Node(
        CollateWithMetadata(
            fields=['effect_maps', 'variance_maps', 'tstat_maps', 'zscore_maps'],
            field_to_metadata_map={
                'effect_maps': {'stat': 'effect'},
                'variance_maps': {'stat': 'variance'},
                #'pvalue_maps': {'stat': 'p'},
                'zscore_maps': {'stat': 'z'},
                'tstat_maps': {'stat' : 't'}
            }),
        name=f'collate_{level}_outputs')

    ds_contrast_maps = pe.Node(
        BIDSDataSink(base_directory=output_dir,
                     path_patterns=image_pattern),
        iterfield=['entities', 'in_file'],
        run_without_submitting=True,
        name=f'ds_{level}_contrast_maps')

    wrangle_outputs = pe.Node(
        IdentityInterface(fields=['contrast_metadata', 'contrast_maps', 'brain_mask']),
        name=f'wrangle_{level}_outputs')

    workflow.connect([
        (wrangle_inputs, get_info, [('contrast_metadata', 'contrast_metadata'),
                                    ('contrast_maps', 'contrast_maps')]),

        (get_info, estimate_model, [('design_matrices', 'design_file'),
                                    ('contrast_matrices', 't_con_file'),
                                    ('covariance_matrices', 'cov_split_file'),
                                    ('dof_maps', 'dof_var_cope_file'),
                                    ('variance_maps', 'var_cope_file'),
                                    ('effect_maps', 'cope_file')]),
        (wrangle_inputs, estimate_model, [('brain_mask', 'mask_file')]),
        (estimate_model, collate, [('copes', 'effect_maps'),
                                   ('var_copes', 'variance_maps'),
                                   ('tstats', 'tstat_maps'),
                                   ('zstats', 'zscore_maps')]),

        (get_info, collate, [('contrast_metadata', 'contrast_metadata')]),

        (collate, collate_outputs, [('effect_maps', 'effect_maps'),
                                    ('variance_maps', 'variance_maps'),
                                    ('tstat_maps', 'tstat_maps'),
                                    ('zscore_maps', 'zscore_maps'),
                                    ('contrast_metadata', 'metadata')]),

        (collate_outputs, ds_contrast_maps, [('out', 'in_file'),
                                             ('metadata', 'entities')]),
        (collate_outputs, wrangle_outputs, [('metadata', 'contrast_metadata'),
                                            ('out', 'contrast_maps')])
    ])

    return workflow
