"""
Helper functions for FSL workflows
"""
# pylint: disable=R0913,R0914,C0415,C0116
# Run Level Functions


def get_btthresh(medianvals):
    return [0.75 * val for val in medianvals]


def get_entities(run_entities, contrasts):
    contrast_entities = []
    contrast_names = [contrast[0] for contrast in contrasts]
    for contrast_name in contrast_names:
        run_entities.update({'contrast': contrast_name})
        contrast_entities.append(run_entities.copy())
    return contrast_entities


def get_usans(central_val):
    return [[tuple([val[0], 0.75 * val[1]])] for val in central_val]


def snake_to_camel(string):
    string.replace('.', '_')
    words = string.replace('.', '').split('_')
    return words[0] + ''.join(word.title() for word in words[1:])


def reshape_ra(run_info, functional_file, outlier_file, contrast_entities):
    import pandas as pd
    import numpy as np
    import nibabel as nb
    from nipype.interfaces.base import Bunch
    run_dict = run_info.dictcopy()
    ntimepoints = nb.load(functional_file).get_data().shape[-1]
    outlier_frame = pd.read_csv(
        outlier_file, header=None, names=['outlier_index'])
    for i, row in outlier_frame.iterrows():
        run_dict['regressor_names'].append(f'rapidart{i:02d}')
        ra_col = np.zeros(ntimepoints)
        ra_col[row['outlier_index']] = 1
        run_dict['regressors'].append(ra_col)
    run_info = Bunch(**run_dict)

    contrast_ents = contrast_entities.copy()
    contrast_entities = []
    for ents in contrast_ents:
        cont_ents = ents.copy()
        curr_dof = cont_ents['DegreesOfFreedom']
        cont_ents.update({'DegreesOfFreedom': curr_dof - len(outlier_frame)})
        contrast_entities.append(cont_ents)
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
    mat_rows, mat_cols = matrix_data.shape
    matrix_path = Path.cwd() / 'run0.mat'
    if matrix_path.is_file():
        open(matrix_path, 'w').close()
    with open(matrix_path, 'a') as dma:
        dma.writelines(
            f'/NumWaves {mat_cols}\t\n/NumPoints {mat_rows}\t\n/Matrix\n')
    for idx, column in matrix_data.T.iterrows():
        if column.max() < .00000000000001:
            matrix_data[idx] = np.ones(len(matrix_data))

    matrix_data.to_csv(
        matrix_path, index=None, columns=None,
        sep='\t', line_terminator='\t\n',
        mode='a', header=None)
    return str(matrix_path)
