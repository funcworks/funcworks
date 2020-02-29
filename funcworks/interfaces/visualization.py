"""
Run and Session Level Visualization interface
"""
#pylint: disable=E1101, C0115
from pathlib import Path
from nipype.interfaces.base import (BaseInterfaceInputSpec,
                                    TraitedSpec, traits, File,
                                    Directory)
from nipype.interfaces.io import IOBase
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from bids.layout.writing import build_path
sns.set_style('white')

class PlotMatricesInputSpec(BaseInterfaceInputSpec):
    run_info = traits.Any(desc='List of regressors of no interest')
    mat_file = File(exists=True, desc='Matrix File produced by Generate Model')
    con_file = File(exists=True, desc='Contrast File Produces by Generate Model')
    entities = traits.Dict(desc='Dictionary containing BIDS file entities')
    output_dir = Directory(desc='Directory for Output')

class PlotMatricesOutputSpec(TraitedSpec):
    design_matrix = traits.Any(desc='Path to design matrix')
    design_plot = File(desc='SVG File containing the plotted Design Matrix')
    contrasts_plot = File(desc='SVG File containing the plotted Contrast Matrix')
    correlation_plot = File(desc='SVG File containing the plotted Correlation Matrix')


class PlotMatrices(IOBase):

    input_spec = PlotMatricesInputSpec
    output_spec = PlotMatricesOutputSpec

    _always_run = True
    def _list_outputs(self):
        ents = self.inputs.entities
        run_info = self.inputs.run_info
        regressor_names = run_info.conditions
        confound_names = run_info.regressor_names
        output_dir = Path(self.inputs.output_dir)
        image_pattern = 'reports/[sub-{subject}/][ses-{session}/]figures/[run-{run:02d}/]' \
            '[sub-{subject}_][ses-{session}_]task-{task}[_acq-{acquisition}]' \
            '[_rec-{reconstruction}][_run-{run}][_echo-{echo}]_' \
            '{suffix<design|corr|contrasts>}.svg'

        design_matrix_pattern = '[sub-{subject}/][ses-{session}/]' \
        '[sub-{subject}_][ses-{session}_]task-{task}[_acq-{acquisition}]' \
        '[_rec-{reconstruction}][_run-{run}][_echo-{echo}]_{suffix<design>}.tsv'

        design_matrix, corr_matrix, contrast_matrix = \
            self._parse_matrices(regressor_names=regressor_names,
                                 confound_names=confound_names,
                                 mat_file=self.inputs.mat_file,
                                 con_file=self.inputs.con_file)
        des_plot = self._plot_matrix(matrix=design_matrix,
                                     entities=ents,
                                     path_pattern=image_pattern,
                                     output_dir=output_dir,
                                     suffix='design')
        con_plot = self._plot_matrix(matrix=contrast_matrix,
                                     entities=ents,
                                     path_pattern=image_pattern,
                                     output_dir=output_dir,
                                     suffix='contrasts')
        corr_plot = self._plot_corr_matrix(corr_matrix=corr_matrix,
                                           entities=ents,
                                           path_pattern=image_pattern,
                                           output_dir=output_dir,
                                           regressor_names=regressor_names)
        ents.update({'suffix':'design'})
        design_path = output_dir / build_path(ents, path_patterns=design_matrix_pattern)
        design_path.parent.mkdir(exist_ok=True, parents=True)
        design_matrix.to_csv(design_path, sep='\t', index=None)
        return {'design_matrix': str(design_path),
                'design_plot': str(des_plot),
                'contrasts_plot': str(con_plot),
                'correlation_plot': str(corr_plot)}

    @staticmethod
    def _parse_matrices(regressor_names, confound_names, mat_file, con_file):
        with open(mat_file, 'r') as matf:
            for idx, line in enumerate(matf.readlines()):
                if '/Matrix' in line:
                    design_matrix = np.genfromtxt(mat_file,
                                                  skip_header=idx + 1,
                                                  delimiter='\t')[:, :-1]
                    break
        contrast_names = []
        with open(con_file, 'r') as matf:
            for idx, line in enumerate(matf.readlines()):
                if 'ContrastName' in line:
                    _, contrast_name = line.split('\t')
                    contrast_names.append(contrast_name.split()[0])
                if '/Matrix' in line:
                    contrast_matrix = np.genfromtxt(con_file, skip_header=idx+1)
                    break
        design_matrix = pd.DataFrame(data=design_matrix, columns=regressor_names + confound_names)
        contrast_matrix = pd.DataFrame(data=contrast_matrix,
                                       columns=regressor_names + confound_names,
                                       index=contrast_names)
        corr_matrix = design_matrix.corr()
        return design_matrix, corr_matrix, contrast_matrix

    @staticmethod
    def _plot_matrix(matrix, entities, path_pattern, output_dir, suffix=None):
        fig = plt.figure(figsize=(14, 10))
        vmax = np.abs(matrix.values).max()
        sns.heatmap(data=matrix, cmap='viridis', ax=fig.gca(),
                    vmin=-vmax, vmax=vmax,
                    cbar_kws={'shrink': 0.5, 'ticks': np.linspace(-vmax, vmax, 5)})
        entities.update({'suffix': suffix})
        matrix_path = output_dir / build_path(entities, path_patterns=path_pattern)
        matrix_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(matrix_path, bbox_inches='tight')
        return matrix_path

    @staticmethod
    def _plot_corr_matrix(corr_matrix, entities, path_pattern,
                          output_dir, regressor_names):
        fig = plt.figure(figsize=(10, 10))
        vmax = np.abs(corr_matrix.values).max()
        plot = sns.heatmap(data=corr_matrix, square=True, cmap='RdBu_r', ax=fig.gca(),
                           vmin=-vmax, vmax=vmax,
                           xticklabels=True, yticklabels=True, linewidths=0.3,
                           cbar_kws={'shrink': 0.5, 'ticks': np.linspace(-vmax, vmax, 5)})
        plot.xaxis.tick_top()
        xtl = plot.get_xticklabels()
        plot.set_xticklabels(xtl, rotation=90)
        plot.hlines([len(regressor_names)], 0, len(regressor_names))
        plot.vlines([len(regressor_names)], 0, len(regressor_names))
        entities.update({'suffix': 'corr'})
        matrix_path = output_dir / build_path(entities, path_patterns=path_pattern)
        matrix_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(matrix_path, bbox_inches='tight')
        return matrix_path
