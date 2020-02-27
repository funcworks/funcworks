"""
Helper functions for FSL workflows
"""
#pylint: disable=R0913,R0914
#Run Level Functions
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
        all_contrasts.append((contrast.split('.')[-1], 'T',
                              [contrast],
                              [1]))
    for contrast in contrasts:
        if not any([all([x in contrast['ConditionList'], y in contrast['ConditionList']]) \
                    for x, y in include_combos])\
        and len(contrast['ConditionList']) == 2:
            continue

        if contrast['Name'] == 'task_vs_baseline':
            weight_vector = [1 * 1 / len(include_contrasts)] * len(include_contrasts)
            all_contrasts.append((contrast['Name'], contrast['Type'].upper(),
                                  include_contrasts,
                                  weight_vector))
        else:
            all_contrasts.append((contrast['Name'], contrast['Type'].upper(),
                                  contrast['ConditionList'],
                                  contrast['Weights']))
    return all_contrasts



def get_smoothing_info_fsl(func, brain_mask, mean_image):
    import nibabel as nb
    import numpy as np
    img = nb.load(func)
    img_data = img.get_fdata()
    mask_img_data = nb.load(brain_mask).get_fdata()
    img_median = np.median(img_data[mask_img_data > 0])
    btthresh = img_median * 0.75
    usans = [tuple([mean_image, btthresh])]

    return usans, btthresh

def get_entities(func_file, contrasts):
    import os #pylint: disable=W0621,W0404
    from bids.layout import parse_file_entities
    contrast_entities = []
    entities = parse_file_entities(func_file)
    contrast_names = [contrast[0] for contrast in contrasts]
    for contrast_name in contrast_names:
        entities.update({'contrast':contrast_name})
        contrast_entities.append(entities.copy())
    return contrast_entities

def snake_to_camel(string):
    string.replace('.', '_')
    words = string.replace('.', '').split('_')
    return words[0] + ''.join(word.title() for word in words[1:])

def rename_outputs(output_dir, contrasts, entities,
                   effects=None, variances=None, zstats=None,
                   pstats=None, tstats=None, fstats=None, dof=None):
    import os #pylint: disable=W0621,W0404
    import subprocess
    import shutil
    from bids.layout.writing import build_path

    stat_dict = dict(effects=effects,
                     variances=variances,
                     z=zstats,
                     t=tstats,
                     F=fstats)
    dof_pattern = ('[sub-{subject}/][ses-{session}/]sub-{subject}'
                   '[_ses-{session}]_task-{task}[_acq-{acquisition}][_rec-{reconstruction}]'
                   '[_run-{run}][_echo-{echo}][_space-{space}]_contrast-{contrast}_dof.tsv')
    contrast_pattern = ('[sub-{subject}/][ses-{session}/]' \
                       '[sub-{subject}_][ses-{session}_]task-{task}[_acq-{acquisition}]'
                       '[_rec-{reconstruction}][_run-{run}][_echo-{echo}][_space-{space}]_'
                       'contrast-{contrast}_stat-{stat<effect|variance|z|p|t|F>}_statmap.nii.gz')

    output_path = os.path.join(output_dir, 'run_level')
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'sub-' + entities["subject"]), exist_ok=True)
    if 'session' in entities:
        os.makedirs(os.path.join(output_path,
                                 'sub-' + entities["subject"],
                                 'ses-' + entities["session"]),
                    exist_ok=True)
    outputs = {'p':[], 'dof':[]}
    contrast_names = [x[0] for x in contrasts]
    for stat, file_list in stat_dict.items():
        outputs[stat] = []
        if not file_list:
            continue
        for idx, file in enumerate(file_list):
            entities.update({'contrast': snake_to_camel(contrast_names[idx]),
                             'stat': stat})
            dest_path = os.path.join(output_path,
                                     build_path(entities, contrast_pattern))
            #os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy(file, dest_path)
            outputs[stat].append(dest_path)
            if stat == 'z':
                entities.update({'stat': 'p'})
                dest_path = os.path.join(output_path,
                                         build_path(entities, contrast_pattern))
                outputs['p'].append(dest_path)
                subprocess.Popen(['fslmaths', f'{file}', '-ztop', f'{dest_path}']).wait()
            #Produce dof file if one doesn't exist for a contrast
            dest_path = os.path.join(output_path,
                                     build_path(entities, dof_pattern))
            if not os.path.isfile(dest_path):
                shutil.copy(dof, dest_path)
                outputs['dof'].append(dest_path)
    effects = outputs['effects']
    variances = outputs['variances']
    zstats = outputs['z']
    pstats = outputs['p']
    tstats = outputs['t']
    fstats = outputs['F']
    dof = outputs['dof']
    return effects, variances, zstats, pstats, tstats, dof, fstats

def reshape_ra(run_info, metadata, outlier_files):
    import os #pylint: disable=W0621,W0404
    import pandas as pd
    import numpy as np
    from nipype.interfaces.base import Bunch
    run_dict = run_info.dictcopy()
    outlier_frame = np.genfromtxt(outlier_files, dtype=int)
    for i, value in enumerate(outlier_frame):
        run_dict['regressor_names'].append(f'rapidart{i:02d}')
        ra_col = np.zeros(metadata['NumTimepoints'])
        ra_col[value] = 1
        run_dict['regressors'].append(ra_col)
    run_info = Bunch(**run_dict)
    return run_info

#Session Level Functions
def num_copes(effects):
    import nibabel as nb
    cope_num = nb.load(effects).shape[-1]
    return cope_num

def get_brainmask(subject_id, derivatives):
    brainmask = (f'{derivatives}/sub-{subject_id}/'
                 f'func/sub-{subject_id}_task-study_run-02_desc-brain_mask.nii.gz')
    return brainmask

def return_contrasts(subject_id, derivatives):
    import os.path as op
    from bids.layout.writing import build_path
    from bids.layout import parse_file_entities
    from glob import glob
    contrasts = []

    run_level_dir = op.join(derivatives.replace('fmriprep', 'funcworks'), 'run_level')
    base_pattern = ('sub-{subject}[/ses-{session}]/sub-{subject}[_ses-{session}]'
                    '_task-{task}_run-??_contrast-{contrast}[_space-{space}]')
    for file in glob((f'{run_level_dir}/sub-{subject_id}/'
                      f'sub-{subject_id}_task-study_run-0?_*_contrast-*.nii.gz')):
        entities = parse_file_entities(file)
        if entities['contrast'] not in contrasts:
            contrasts.append(entities['contrast'])
    contrast_dof = {}
    contrast_file = {}
    contrast_variances = {}
    for contrast in contrasts:
        entities.update({'contrast':contrast})
        cope_path = op.join(run_level_dir,
                            build_path(entities,
                                       path_patterns=(base_pattern +
                                                      '_stat-effects_statmap.nii.gz')))
        dof_path = op.join(run_level_dir,
                           build_path(entities,
                                      path_patterns=(base_pattern + '_dof.tsv')))
        variances_path = op.join(run_level_dir,
                                 build_path(entities,
                                            path_patterns=(base_pattern +
                                                           '_stat-variances_statmap.nii.gz')))
        contrast_dof[contrast] = sorted(glob(dof_path))
        contrast_file[contrast] = sorted(glob(cope_path))
        contrast_variances[contrast] = sorted(glob(variances_path))

    dofs = [contrast_dof[x] for x in contrast_dof]
    effects = [contrast_file[x] for x in contrast_file]
    variances = [contrast_variances[x] for x in contrast_variances]

    return effects, variances, dofs


def merge_runs(effects, variances, dofs):
    import os #pylint: disable=W0621,W0404
    import os.path as op
    from bids.layout.writing import build_path
    import numpy as np
    import nibabel as nb
    from bids.layout import parse_file_entities
    full_effects = np.concatenate([np.expand_dims(nb.load(x).get_fdata(), 3) \
                                   for x in effects], axis=3)
    full_variances = np.concatenate([np.expand_dims(nb.load(x).get_fdata(), 3) \
                                     for x in variances], axis=3)
    full_dofs = np.ones_like(full_effects)
    full_dofs = np.concatenate([full_dofs[:, :, :, i] * np.loadtxt(file) \
                                for i, file in enumerate(dofs)])
    affine = nb.load(effects[0]).affine
    entities = parse_file_entities(effects[0])

    effects_path = op.join(os.getcwd(),
                           build_path(entities,
                                      path_patterns=('sub-{subject}[_ses-{session}]'
                                                     '_task-{task}_contrast-{contrast}'
                                                     '_stat-effects_merged.nii.gz')))
    nb.save(nb.nifti1.Nifti1Image(full_variances, affine), effects_path)
    variances_path = op.join(os.getcwd(),
                             build_path(entities,
                                        path_patterns=('sub-{subject}[_ses-{session}]'
                                                       '_task-{task}_contrast-{contrast}'
                                                       '_stat-variances_merged.nii.gz')))
    nb.save(nb.nifti1.Nifti1Image(full_variances, affine), variances_path)
    dofs_path = op.join(os.getcwd(),
                        build_path(entities,
                                   path_patterns=('sub-{subject}[_ses-{session}]'
                                                  '_task-{task}_contrast-{contrast}'
                                                  '_stat-dof_merged.nii.gz')))
    nb.save(nb.nifti1.Nifti1Image(full_dofs, affine), dofs_path)
    return effects_path, variances_path, dofs_path


def rename_contrasts(effects, variances, tstats, zstats, res4d):
    import os.path as op
    from bids.layout.writing import build_path
    from bids.layout import parse_file_entities

    if not isinstance(effects, list):
        effects = [effects]
    if not isinstance(variances, list):
        variances = [variances]
    if not isinstance(tstats, list):
        tstats = [tstats]
    if not isinstance(zstats, list):
        zstats = [zstats]
    if not isinstance(res4d, list):
        res4d = [res4d]

    entities = parse_file_entities(effects[0])
    new_names = []
    for i, _ in enumerate(effects):
        new_names.append((op.basename(effects[i]),
                          build_path(entities,
                                     path_patterns=('sub-{subject}[_ses-{sesion}]'
                                                    '_task-{task}_contrast-{contrast}'
                                                    '_stat-effects_statmap.nii.gz'))))
        new_names.append((op.basename(variances[i]),
                          build_path(entities,
                                     path_patterns=('sub-{subject}[_ses-{session}]'
                                                    '_task-{task}_contrast-{contrast}'
                                                    '_stat-variances_statmap.nii.gz'))))
        new_names.append((op.basename(tstats[i]),
                          build_path(entities,
                                     path_patterns=('sub-{subject}[_ses-{session}]'
                                                    '_task-{task}_contrast-{contrast}'
                                                    '_stat-t_statmap.nii.gz'))))
        new_names.append((op.basename(zstats[i]),
                          build_path(entities,
                                     path_patterns=('sub-{subject}[_ses-{session}]'
                                                    '_task-{task}_contrast-{contrast}'
                                                    '_stat-z_statmap.nii.gz'))))
        new_names.append((op.basename(res4d[i]),
                          build_path(entities,
                                     path_patterns=('sub-{subject}[_ses-{session}]'
                                                    '_task-{task}_contrast-{contrast}'
                                                    '_stat-residuals_statmap.nii.gz'))))
    return new_names
