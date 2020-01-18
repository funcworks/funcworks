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
        all_contrasts.append((contrast, 'T',
                              [contrast.split('.')[-1]],
                              [1]))
    for contrast in contrasts:
        if not any([all([x in contrast['ConditionList'], y in contrast['ConditionList']]) \
                    for x, y in include_combos])\
        and len(contrast['ConditionList']) == 2:
            continue

        if contrast['Name'] == 'task_vs_baseline':
            condition_list = [x.split('.')[-1] if '.' in x else x for x in include_contrasts]
            weight_vector = [1 * 1 / len(include_contrasts)] * len(include_contrasts)
            all_contrasts.append((contrast['Name'], contrast['Type'].upper(),
                                  condition_list,
                                  weight_vector))
        else:
            condition_list = \
            [x.split('.')[-1] if '.' in x else x for x in contrast['ConditionList']]
            all_contrasts.append((contrast['Name'], contrast['Type'].upper(),
                                  condition_list,
                                  contrast['Weights']))
    return all_contrasts

def get_info(confounds, events, confound_regressors, condition_names):
    '''Grabs EV files for subject based on contrasts of interest'''
    from nipype.interfaces.base import Bunch
    import pandas as pd
    import numpy as np
    event_data = pd.read_csv(events, sep='\t')
    conf_data = pd.read_csv(confounds, sep='\t')
    conf_data = conf_data.fillna(0)
    names = []
    onsets = []
    amplitudes = []
    durations = []
    regressor_names = []
    regressors = []
    for condition in condition_names:
        try:
            condition_column, condition_name = condition.split('.')
        except ValueError:
            print(('Name of Conditions must consist of a'
                   'string of the format {column_name}.{trial_name}'
                   f'but a value of "{condition}" was specified'))
        condition_frame = event_data.query(f'{condition_column} == "{condition_name}"')
        if len(condition_frame) > 0:
            names.append(condition)
            onsets.append(condition_frame['onset'].values)
            durations.append(condition_frame['duration'].values)
            amplitudes.append(np.ones(len(condition_frame)))

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
    #A workaround to a current issue in pybids
    #that causes massive resource use when indexing derivative tsv files
    lead = func.split('desc')[0]
    confound_file = lead + 'desc-confounds_regressors.tsv'
    return confound_file

def get_metadata(func):
    import json
    import nibabel as nb
    num_timepoints = nb.load(func).get_data().shape[3]
    lead = func.split('.nii.gz')[0]
    metafile = lead + '.json'
    with open(metafile) as omf:
        metadata = json.load(omf)

    return metadata['RepetitionTime'], num_timepoints

def get_motion_parameters(confounds):
    import os #pylint: disable=W0621,W0404
    import pandas as pd
    motion_params = os.path.join(os.getcwd(),
                                 os.path.basename(confounds).split('.')[0] + '_motparams.tsv')
    confound_data = pd.read_csv(confounds, sep='\t')
    #Motion data gets formatted FSL style, with x, y, z rotation, then x,y,z translation
    motion_data = confound_data[['rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z']]
    motion_data.to_csv(motion_params, sep='\t', header=None, index=None)
    return motion_params

def get_smoothing_info_fsl(func, brain_mask, mean_image):
    import nibabel as nb
    import numpy as np
    img = nb.load(func)
    img_data = img.get_data()
    mask_img_data = nb.load(brain_mask).get_data()
    img_median = np.median(img_data[mask_img_data > 0])
    btthresh = img_median * 0.75
    usans = [tuple([mean_image, btthresh])]

    return usans, btthresh

def get_entities(func):
    import os #pylint: disable=W0621,W0404
    run_entities = []
    for func_file in func:
        stem = os.path.basename(func_file).split('.')[0]
        entity_pairs = stem.split('_')
        entities = {x.split('-')[0]:x.split('-')[1] if '-' in x else None for x in entity_pairs}
        for item in entities:
            if entities[item] is None:
                entities['suffix'] = item
                break
        entities['subject'] = entities.pop('sub', None)
        run_entities.append(entities)
    return run_entities

def rename_outputs(bids_dir, output_dir, contrasts, entities,
                   effects=None, variances=None, zstats=None, pstats=None, tstats=None, fstats=None, dof=None):
    import os #pylint: disable=W0621,W0404
    import subprocess
    import shutil
    from bids import BIDSLayout
    def snake_to_camel(string):
        string.replace('.', '_')
        words = string.replace('.', '').split('_')
        return words[0] + ''.join(word.title() for word in words[1:])
    stat_dict = dict(effects=effects,
                     variances=variances,
                     zstats=zstats,
                     tstats=tstats,
                     fstats=fstats)
    dof_pattern = ('[sub-{subject}/][ses-{session}/]sub-{subject}'
                   '[_ses-{session}]_task-{task}[_acq-{acquisition}][_rec-{reconstruction}]'
                   '[_run-{run}][_echo-{echo}][_space-{space}]_contrast-{contrast}_dof.tsv')
    contrast_pattern = ('[sub-{subject}/][ses-{session}/]' \
                       '[sub-{subject}_][ses-{session}_]task-{task}[_acq-{acquisition}]'
                       '[_rec-{reconstruction}][_run-{run}][_echo-{echo}][_space-{space}]_'
                       'contrast-{contrast}_stat-{stat<effect|variance|z|p|t|F>}_statmap.nii.gz')
    layout = BIDSLayout(bids_dir, validate=False)

    output_path = os.path.join(output_dir, 'run_level')
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'sub-' + entities["subject"]), exist_ok=True)
    if 'session' in entities:
        os.makedirs(os.path.join(output_path,
                                 'sub-' + entities["subject"],
                                 'ses-' + entities["session"]),
                    exist_ok=True)
    outputs = {'pstats':[], 'dof':[]}
    contrast_names = [x[0] for x in contrasts]
    for stat, file_list in stat_dict.items():
        outputs[stat] = []
        if isinstance(file_list, None):
            continue
        for idx, file in enumerate(file_list):
            entities['contrast'] = snake_to_camel(contrast_names[idx])
            entities['stat'] = stat
            dest_path = os.path.join(output_path,
                                     layout.build_path(entities, contrast_pattern, validate=False))
            #os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy(file, dest_path)
            outputs[stat].append(dest_path)
            if stat == 'z':
                entities['stat'] = 'p'
                dest_path = os.path.join(output_path,
                                         layout.build_path(entities,
                                                           contrast_pattern,
                                                           validate=False))
                outputs['p'].append(dest_path)
                subprocess.Popen(['fslmaths', f'{file}', '-ztop', f'{dest_path}']).wait()
            #Produce dof file if one doesn't exist for a contrast
            dest_path = os.path.join(output_path,
                                     layout.build_path(entities, dof_pattern, validate=False))
            if not os.path.isfile(dest_path):
                shutil.copy(dof, dest_path)
                outputs['dof'].append(dest_path)
    effects = outputs['effects']
    variances = outputs['variances']
    zstats = outputs['zstats']
    pstats = outputs['pstats']
    tstats = outputs['tstats']
    fstats = outputs['fstats']
    dof = outputs['dof']
    return effects, variances, zstats, pstats, tstats, dof, fstats

def reshape_ra(outlier_files, confounds, confound_regressors, num_timepoints):
    import os #pylint: disable=W0621,W0404
    import pandas as pd
    import numpy as np

    art_dict = {}
    outlier_frame = pd.read_csv(outlier_files, header=None)
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
    from bids import BIDSLayout
    from glob import glob
    contrasts = []

    run_level_dir = op.join(derivatives.replace('fmriprep', 'funcworks'), 'run_level')
    layout = BIDSLayout(run_level_dir, validate=False)

    for file in glob((f'{run_level_dir}/sub-{subject_id}/'
                      f'sub-{subject_id}_task-study_run-0?_contrast-*.nii.gz')):
        entities = {pair.split('-')[0]:pair.split('-')[1] \
                    for pair in op.basename(file).split('_') if '-' in pair}
        if entities['contrast'] not in contrasts:
            contrasts.append(entities['contrast'])
    contrast_dof = {}
    contrast_file = {}
    contrast_variances = {}
    for contrast in contrasts:
        entities['contrast'] = contrast
        cope_path = op.join(run_level_dir,
                            layout.build_path(entities,
                                              path_patterns=('sub-{sub}[/ses-{ses}]/sub-{sub}[_ses-{ses}]'
                                                             '_task-{task}_run-??_contrast-{contrast}'
                                                             '_stat-effects_statmap.nii.gz'), validate=False))
        dof_path = op.join(run_level_dir,
                           layout.build_path(entities,
                                             path_patterns=('sub-{sub}[/ses-{ses}]/sub-{sub}[_ses-{ses}]'
                                                            '_task-{task}_run-??_contrast-{contrast}'
                                                            '_dof.tsv'), validate=False))
        variances_path = op.join(run_level_dir,
                                 layout.build_path(entities,
                                                   path_patterns=('sub-{sub}[/ses-{ses}]/sub-{sub}[_ses-{ses}]'
                                                                  '_task-{task}_run-??_contrast-{contrast}'
                                                                  '_stat-variances_statmap.nii.gz'), validate=False))
        contrast_dof[contrast] = sorted(glob(dof_path))
        contrast_file[contrast] = sorted(glob(cope_path))
        contrast_variances[contrast] = sorted(glob(variances_path))

    dofs = [contrast_dof[x] for x in contrast_dof]
    effects = [contrast_file[x] for x in contrast_file]
    variances = [contrast_variances[x] for x in contrast_variances]

    return effects, variances, dofs


def merge_runs(effects, variances, dofs, derivatives):
    import os #pylint: disable=W0621,W0404
    import os.path as op
    from bids import BIDSLayout
    import numpy as np
    import nibabel as nb
    run_level_dir = os.path.join(derivatives.replace('fmriprep', 'funcworks'), 'run_level')
    layout = BIDSLayout(run_level_dir, validate=False)

    full_effects = np.concatenate([np.expand_dims(nb.load(x).get_data(), 3) \
                                   for x in effects], axis=3)
    full_variances = np.concatenate([np.expand_dims(nb.load(x).get_data(), 3) \
                                     for x in variances], axis=3)
    full_dofs = np.ones_like(full_effects)
    full_dofs = np.concatenate([full_dofs[:, :, :, i] * np.loadtxt(file) \
                                for i, file in enumerate(dofs)])
    affine = nb.load(effects[0]).affine
    entities = {pair.split('-')[0]:pair.split('-')[1] \
                for pair in os.path.basename(effects[0]).split('_') if '-' in pair}
    effects_img = nb.nifti1.Nifti1Image(full_variances, affine)
    variances_img = nb.nifti1.Nifti1Image(full_variances, affine)
    dofs_img = nb.nifti1.Nifti1Image(full_dofs, affine)
    effects_path = op.join(os.getcwd(),
                           layout.build_path(entities,
                                             path_patterns=('sub-{sub}[_ses-{ses}]'
                                                            '_task-{task}_contrast-{contrast}'
                                                            '_stat-effects_merged.nii.gz'),
                                             validate=False))
    nb.save(effects_img, effects_path)
    variances_path = op.join(os.getcwd(),
                             layout.build_path(entities,
                                               path_patterns=('sub-{sub}[_ses-{ses}]'
                                                              '_task-{task}_contrast-{contrast}'
                                                              '_stat-variances_merged.nii.gz'),
                                               validate=False))
    nb.save(variances_img, variances_path)
    dofs_path = op.join(os.getcwd(),
                        layout.build_path(entities,
                                          path_patterns=('sub-{sub}[_ses-{ses}]'
                                                         '_task-{task}_contrast-{contrast}'
                                                         '_stat-dof_merged.nii.gz'),
                                          validate=False))
    nb.save(dofs_img, dofs_path)
    return effects_path, variances_path, dofs_path


def rename_contrasts(merged_effects, effects, variances, tstats, zstats, res4d):
    import os.path as op
    from bids import BIDSLayout
    layout = BIDSLayout('', validate=False)
    if not isinstance(merged_effects, list):
        merged_effects = [merged_effects]
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
    entities = {pair.split('-')[0]:pair.split('-')[1] \
                for pair in op.basename(merged_effects[0]).split('_') if '-' in pair}
    new_names = []
    for i, _ in enumerate(merged_effects):
        new_names.append((op.basename(effects[i]),
                          layout.build_path(entities,
                                            path_patterns=('sub-{sub}[_ses-{ses}]'
                                                           '_task-{task}_contrast-{contrast}'
                                                           '_stat-effects_statmap.nii.gz'),
                                            validate=False)))
        new_names.append((op.basename(variances[i]),
                          layout.build_path(entities,
                                            path_patterns=('sub-{sub}[_ses-{ses}]'
                                                           '_task-{task}_contrast-{contrast}'
                                                           '_stat-variances_statmap.nii.gz'),
                                            validate=False)))
        new_names.append((op.basename(tstats[i]),
                          layout.build_path(entities,
                                            path_patterns=('sub-{sub}[_ses-{ses}]'
                                                           '_task-{task}_contrast-{contrast}'
                                                           '_stat-t_statmap.nii.gz'),
                                            validate=False)))
        new_names.append((op.basename(zstats[i]),
                          layout.build_path(entities,
                                            path_patterns=('sub-{sub}[_ses-{ses}]'
                                                           '_task-{task}_contrast-{contrast}'
                                                           '_stat-z_statmap.nii.gz'),
                                            validate=False)))
        new_names.append((op.basename(res4d[i]),
                          layout.build_path(entities,
                                            path_patterns=('sub-{sub}[_ses-{ses}]'
                                                           '_task-{task}_contrast-{contrast}'
                                                           '_stat-residuals_statmap.nii.gz'),
                                            validate=False)))
    return new_names
