"""
Helper functions for FSL workflows
"""
#pylint: disable=R0913,R0914,C0415,C0116
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

def get_entities(run_entities, contrasts):
    contrast_entities = []
    contrast_names = [contrast[0] for contrast in contrasts]
    for contrast_name in contrast_names:
        run_entities.update({'contrast':contrast_name})
        contrast_entities.append(run_entities.copy())
    return contrast_entities

def snake_to_camel(string):
    string.replace('.', '_')
    words = string.replace('.', '').split('_')
    return words[0] + ''.join(word.title() for word in words[1:])

def reshape_ra(run_info, func, outlier_files, contrast_entities):
    import pandas as pd
    import numpy as np
    import nibabel as nb
    from nipype.interfaces.base import Bunch
    run_dict = run_info.dictcopy()
    ntimepoints = nb.load(func).get_data().shape[-1]
    outlier_frame = pd.read_csv(outlier_files, header=None, names=['outlier_index'])
    for i, row in outlier_frame.iterrows():
        run_dict['regressor_names'].append(f'rapidart{i:02d}')
        ra_col = np.zeros(ntimepoints)
        ra_col[row['outlier_index']] = 1
        run_dict['regressors'].append(ra_col)
    run_info = Bunch(**run_dict)
    contrast_ents = contrast_entities.copy()
    contrast_entities = []
    for contrast_ents in contrast_ents:
        curr_dof = contrast_ents['DegreesOfFreedom'].values
        contrast_ents.update({
            'DegreesOfFreedom': (curr_dof - len(outlier_frame['outlier_index']))
        })
        contrast_entities.append(contrast_ents)
    return run_info, contrast_entities

def correct_matrix(design_matrix):
    import numpy as np
    import pandas as pd
    from pathlib import Path

    with open(design_matrix, 'r') as dmr:
        content = dmr.readlines()
        matrix_index = content.index('/Matrix\n') + 1
    matrix_data = pd.read_csv(
        design_matrix, skiprows=matrix_index, delim_whitespace=True,
        header=None)
    matrix_path = Path.cwd() / 'run0.mat'
    if matrix_path.is_file():
        open(matrix_path, 'w').close()
    with open(matrix_path, 'a') as dma:
        dma.writelines('/NumWaves {columns}\t\n'.format(columns=matrix_data.shape[1]))
        dma.writelines('/NumPoints {rows}\t\n'.format(rows=matrix_data.shape[0]))
        dma.writelines('/Matrix\n')
    for idx, column in matrix_data.T.iterrows():
        if column.max() < .00000000000001:
            matrix_data[idx] = np.ones(len(matrix_data))

    matrix_data.to_csv(
        matrix_path, index=None, columns=None,
        sep='\t', line_terminator='\t\n',
        mode='a', header=None)
    return str(matrix_path)

def reference_outputs(**args):
    def _pop(inlist):
        if isinstance(inlist, (list, tuple)) and len(inlist) == 1:
            return inlist[0]
        return inlist
    popped_lists = {}
    for arg in args:
        popped_lists[arg] = _pop(args[arg])
    return popped_lists['brain_mask'], popped_lists['bold_ref']
