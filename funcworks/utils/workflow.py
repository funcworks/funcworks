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

def get_entities(func_file, contrasts):
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

def reshape_ra(run_info, metadata, outlier_files):
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
