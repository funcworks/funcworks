"""Tests for utils."""
import numpy as np
from funcworks import utils
from pathlib import Path


def test__get_btthresh():
    """Test get_btthresh."""
    median_vals = np.arange(0, 10, 0.1).tolist()
    output = utils.get_btthresh(median_vals)
    assert len(median_vals) == len(output)


def test__get_usans():
    """Test get_usans."""
    median_vals = np.arange(0, 10, 0.1).tolist()
    mean_vals = np.arange(0, 10, 0.1).tolist()
    output = utils.get_usans(zip(median_vals, mean_vals))
    assert len(median_vals) == len(output)


def test__snake_to_camel():
    """Test snake_to_camel."""
    expected = "trialTypenegLureFa"
    input = "trial_type.neg_lure_fa"
    output = utils.snake_to_camel(input)
    assert output == expected


def test__reshape_ra():
    """Test reshape_ra."""
    from nipype.interfaces.base import Bunch
    import nibabel as nb

    run_info = Bunch(**{"regressors": [], "regressor_names": []})
    outlier_file = Path.cwd() / "outlier_test.txt"
    np.savetxt(outlier_file, np.array([[0], [1], [36], [54], [60], [75]]))
    test_img = nb.nifti1.Nifti1Image(np.ones((90, 90, 90, 90)), np.eye(4))
    nb.save(test_img, Path.cwd() / "test.nii.gz")
    contrast_entities = [{"DegreesOfFreedom": 9}]
    (output_run_info, output_contrast_entities) = utils.reshape_ra(
        run_info, Path.cwd() / "test.nii.gz", outlier_file, contrast_entities
    )
    for contrast_ents in output_contrast_entities:
        assert contrast_ents["DegreesOfFreedom"] == 3
    assert len(output_run_info.regressors) == 6
    for col in output_run_info.regressors:
        assert np.sum(col) == 1


def test__flatten():
    """Test flatten."""
    input = [[1], [2], [3]]
    expected = [1, 2, 3]
    output = utils.flatten(input)
    assert output == expected


def test__correct_matrices():
    """Test correct_matrices."""
    import filecmp

    curr_dir = Path(__file__).parent
    matrix_file = curr_dir / "data" / "run0.mat"
    output = utils.correct_matrix(matrix_file)
    assert filecmp.cmp(output, curr_dir / "data" / "run0_corrected.mat")
