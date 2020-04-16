"""Run and Session Level Visualization interface."""
from pathlib import Path
from nipype.interfaces.base import (BaseInterfaceInputSpec,
                                    TraitedSpec, traits, File,
                                    Directory)
from nipype.interfaces.io import IOBase
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from bids import BIDSLayout
sns.set_style('white')


class _PlotMatricesInputSpec(BaseInterfaceInputSpec):
    run_info = traits.Any(desc='List of regressors of no interest')
    mat_file = File(
        exists=True,
        desc='Matrix File produced by Generate Model')
    con_file = File(
        exists=True,
        desc='Contrast File Produces by Generate Model')
    database_path = Directory(
        exists=True,
        desc='Database path for current model')
    entities = traits.Dict(desc='Dictionary containing BIDS file entities')
    output_dir = Directory(desc='Directory for Output')


class _PlotMatricesOutputSpec(TraitedSpec):
    design_matrix = traits.Any(desc='Path to design matrix')
    design_plot = File(
        desc='SVG File containing the plotted Design Matrix')
    contrasts_plot = File(
        desc='SVG File containing the plotted Contrast Matrix')
    correlation_plot = File(
        desc='SVG File containing the plotted Correlation Matrix')


class PlotMatrices(IOBase):
    """Plot matrices for a given design."""

    input_spec = _PlotMatricesInputSpec
    output_spec = _PlotMatricesOutputSpec
    _always_run = True

    def _list_outputs(self):
        ents = self.inputs.entities
        run_info = self.inputs.run_info
        regressor_names = run_info.conditions
        confound_names = run_info.regressor_names
        output_dir = Path(self.inputs.output_dir)
        layout = BIDSLayout.load(self.inputs.database_path)
        image_pattern = ('reports/[sub-{subject}/][ses-{session}/]'
                         'figures/[run-{run}/]'
                         '[sub-{subject}_][ses-{session}_]'
                         'task-{task}[_acq-{acquisition}]'
                         '[_rec-{reconstruction}][_run-{run}][_echo-{echo}]_'
                         '{suffix<design|corr|contrasts>}.svg')

        design_matrix_patt = ('[sub-{subject}/][ses-{session}/]'
                              '[sub-{subject}_][ses-{session}_]'
                              'task-{task}_[acq-{acquisition}_]'
                              '[rec-{reconstruction}_][run-{run}_]'
                              '[echo-{echo}_]{suffix<design>}.tsv')

        (design_matrix, corr_matrix,
         contrast_matrix) = self._parse_matrices(
            regressor_names=regressor_names,
            confound_names=confound_names,
            mat_file=self.inputs.mat_file,
            con_file=self.inputs.con_filet)
        des_plot = self._plot_matrix(matrix=design_matrix,
                                     path_pattern=image_pattern,
                                     suffix='design',
                                     cmap='viridis')
        con_plot = self._plot_matrix(matrix=contrast_matrix,
                                     path_pattern=image_pattern,
                                     suffix='contrasts',
                                     cmap='RdBu_r',
                                     layout=layout)
        corr_plot = self._plot_corr_matrix(corr_matrix=corr_matrix,
                                           path_pattern=image_pattern,
                                           regressor_names=regressor_names,
                                           cmap='RdBu_r',
                                           layout=layout)
        ents.update({'suffix': 'design'})
        design_path = layout.build_path(ents, path_patterns=design_matrix_patt)
        design_path = output_dir / design_path
        design_path.parent.mkdir(exist_ok=True, parents=True)
        design_matrix.to_csv(design_path, sep='\t', index=None)

        return {'design_matrix': str(design_path),
                'design_plot': str(des_plot),
                'contrasts_plot': str(con_plot),
                'correlation_plot': str(corr_plot)}

    @staticmethod
    def _parse_matrices(regressor_names, confound_names, mat_file, con_file):
        with open(mat_file, 'r') as matf:
            content = matf.readlines()
        design_matrix = pd.read_csv(
            mat_file, skiprows=content.index('/Matrix\n') + 1,
            delim_whitespace=True, header=None)
        with open(con_file, 'r') as matf:
            content = matf.readlines()
        contrast_names = [x for x in content if 'ContrastName' in x]
        contrast_names = [x.split('\t')[-1] for x in contrast_names]
        contrast_matrix = pd.read_csv(
            con_file, skiprows=content.index('/Matrix\n') + 1,
            delim_whitespace=True, header=None)

        design_matrix.columns = regressor_names + confound_names
        contrast_matrix.columns = regressor_names + confound_names
        contrast_matrix.index = contrast_names
        corr_matrix = design_matrix.corr()

        return design_matrix, corr_matrix, contrast_matrix

    def _plot_matrix(self, matrix, path_pattern, suffix=None,
                     cmap='viridis', layout=None):
        fig = plt.figure(figsize=(14, 10))
        vmax = np.abs(matrix.values).max()
        sns.heatmap(
            data=matrix, cmap=cmap, ax=fig.gca(),
            vmin=-vmax, vmax=vmax,
            cbar_kws={'shrink': 0.5, 'ticks': np.linspace(-vmax, vmax, 5)})
        entities = self.inputs.entities
        entities.update({'suffix': suffix})
        fig_path = layout.build_path(entities, path_patterns=path_pattern)
        fig_path = Path(self.inputs.output_dir) / fig_path
        fig_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()

        return fig_path

    def _plot_corr_matrix(self, corr_matrix, path_pattern,
                          regressor_names, cmap=None, layout=None):
        fig = plt.figure(figsize=(10, 10))
        plot = sns.heatmap(
            data=corr_matrix, square=True, cmap=cmap, ax=fig.gca(),
            vmin=-1, vmax=1, xticklabels=True, yticklabels=True,
            linewidths=0.3,
            cbar_kws={'shrink': 0.5, 'ticks': np.linspace(-1, 1, 5)})
        plot.xaxis.tick_top()
        xtl = plot.get_xticklabels()
        plot.set_xticklabels(xtl, rotation=90)
        plot.hlines([len(regressor_names)], 0, len(regressor_names))
        plot.vlines([len(regressor_names)], 0, len(regressor_names))
        entities = self.inputs.entities
        entities.update({'suffix': 'corr'})
        fig_path = layout.build_path(entities, path_patterns=path_pattern)
        fig_path = Path(self.inputs.output_dir) / fig_path
        fig_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()

        return fig_path
