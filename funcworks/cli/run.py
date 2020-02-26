import os
import gc
import sys
import uuid
import json
import logging
import warnings
from pathlib import Path
from tempfile import mkdtemp
from time import strftime
from multiprocessing import cpu_count
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
#pylint: disable=C0103,R0913,R0914,W0404

logging.addLevelName(25, 'IMPORTANT')  # Add a new level between INFO and WARNING
logging.addLevelName(15, 'VERBOSE')  # Add a new level between INFO and DEBUG
logger = logging.getLogger('cli')

def check_deps(workflow):
    from nipype.utils.filemanip import which
    return sorted(
        (node.interface.__class__.__name__, node.interface._cmd)
        for node in workflow._get_all_nodes()
        if (hasattr(node.interface, '_cmd') and
            which(node.interface._cmd.split()[0]) is None))

def _warn_redirect(message, category, filename, lineno, file=None, line=None):
    logger.warning('Captured warning (%s): %s', category, message)

def get_parser():
    """Build Parser Object"""
    parser = ArgumentParser(description='FUNCWORKs: fMRI FUNCtional WORKflows',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('bids_dir', action='store', type=Path,
                        help='the root folder of a BIDS valid dataset (sub-XXXXX folders should '
                             'be found at the top level in this folder).')
    parser.add_argument('output_dir', action='store', type=Path,
                        help='the output path for the outcomes of preprocessing and visual '
                             'reports')
    parser.add_argument('analysis_level', choices=['run', 'session', 'participant', 'dataset'],
                        help='processing stage to be runa (see BIDS-Apps specification).')
    parser.add_argument('-m', '--model-file', action='store', type=Path,
                        help='location of BIDS model description')
    parser.add_argument('-d', '--derivatives', action='store', nargs='+',
                        help='location of derivatives (including preprocessed images).'
                        'If none specified, indexes all derivatives under bids_dir/derivatives.')
    parser.add_argument('--participant_label', '--participant-label', action='store', nargs='+',
                        help='a space delimited list of participant identifiers or a single '
                             'identifier (the sub- prefix can be removed)')
    parser.add_argument('-s', '--smoothing', action='store', metavar="FWHM[:LEVEL:[TYPE]]",
                        default=None,
                        help="Smooth BOLD series with FWHM mm kernel prior to fitting at LEVEL. "
                             "Optional analysis LEVEL (default: l1) may be specified numerically "
                             "(e.g., `l1`) or by name (`run`, `subject`, `session` or `dataset`). "
                             "Optional smoothing TYPE (default: iso) must be one of: `iso` (isotropic). "
                             "e.g., `--smoothing 5:dataset:iso` will perform a 5mm FWHM isotropic "
                             "smoothing on subject-level maps before evaluating the dataset level.")
    parser.add_argument('-w', '--work_dir', action='store', type=Path,
                        default=mkdtemp(),
                        help='path where intermediate results should be stored')
    parser.add_argument('--use-rapidart', action='store_true', default=False,
                        help='Use RapidArt artifact detection algorithm')
    parser.add_argument('--use-plugin', action='store', default=None,
                        help='nipype plugin configuration file')
    return parser

def main():
    """Entry Point"""
    from nipype import logging as nlogging
    from multiprocessing import set_start_method, Process, Manager
    set_start_method('forkserver')
    warnings.showwarning = _warn_redirect

    opts = get_parser().parse_args()
    exec_env = os.name

    sentry_sdk = None
    #if not opts.notrack:
    #    import sentry_sdk
    #    from ..utils.sentry import sentry_setup
    #    sentry_setup(opts, exec_env)

    if opts.analysis_level not in ['run', 'session', 'participant', 'dataset']:
        raise ValueError((f'Unknown analysis level {opts.analysis_level}',
                          "analysis level must be 'run', 'session', 'participant', 'dataset'"))
    if opts.analysis_level not in ['run']:
        raise NotImplementedError((f'{opts.analysis_level} not yet implemented'))
    with Manager() as mgr:
        retval = mgr.dict()

        p = Process(target=build_workflow, args=(opts, retval))
        p.start()
        p.join()

        retcode = p.exitcode or retval.get('return_code', 0)

        bids_dir = retval.get('bids_dir')
        output_dir = retval.get('output_dir')
        work_dir = retval.get('work_dir')
        subject_list = retval.get('participant_label', None)
        funcworks_wf = retval.get('workflow', None)
        run_uuid = retval.get('run_uuid', None)
        plugin_settings = retval.get('plugin_settings')

    retcode = retcode or int(funcworks_wf is None)
    if retcode != 0:
        sys.exit(retcode)

    missing = check_deps(funcworks_wf)
    if missing:
        print("Cannot run FUNCWorks. Missing dependencies:", file=sys.stderr)
        for iface, cmd in missing:
            print("\t{} (Interface: {})".format(cmd, iface))
        sys.exit(2)
    # Clean up master process before running workflow, which may create forks
    gc.collect()

    errno = 1  # Default is error exit unless otherwise set
    funcworks_wf.write_graph(graph2use="colored", format='png')
    try:
        funcworks_wf.run(**plugin_settings)
    except Exception as e:
        #if not opts.notrack:
        #    from ..utils.sentry import process_crashfile
        #    crashfolders = [output_dir / 'funcworks' / 'sub-{}'.format(s) / 'log' / run_uuid
        #                    for s in subject_list]
        #   for crashfolder in crashfolders:
        #        for crashfile in crashfolder.glob('crash*.*'):
        #            process_crashfile(crashfile)
        #
        #   if "Workflow did not execute cleanly" not in str(e):
        #        sentry_sdk.capture_exception(e)
        logger.critical('FUNCWorks failed: %s', e)
        raise

def build_workflow(opts, retval):
    """
    Create the Nipype Workflow that supports the whole execution
    graph, given the inputs.
    All the checks and the construction of the workflow are done
    inside this function that has pickleable inputs and output
    dictionary (``retval``) to allow isolation using a
    ``multiprocessing.Process`` that allows funcworks to enforce
    a hard-limited memory-scope.
    """
    from bids import BIDSLayout

    from nipype import logging as nlogging, config as ncfg
    #from ..__about__ import __version__
    from ..workflows.base import init_funcworks_wf
    __version__ = '0.0.1'
    build_log = nlogging.getLogger('nipype.workflow')

    INIT_MSG = """
    Running FUNCWORKS version {version}:
      * BIDS dataset path: {bids_dir}.
      * Participant list: {participant_label}.
      * Run identifier: {uuid}.
    """.format
    output_dir = opts.output_dir.resolve()
    bids_dir = opts.bids_dir.resolve()
    work_dir = opts.work_dir.resolve()
    retval['return_code'] = 1
    retval['workflow'] = None
    retval['bids_dir'] = bids_dir
    retval['output_dir'] = output_dir
    retval['work_dir'] = work_dir

    if output_dir == bids_dir:
        build_log.error(
            'The selected output folder is the same as the input BIDS folder. '
            'Please modify the output path (suggestion: %s).',
            bids_dir / 'derivatives' / ('funcworks-%s' % __version__.split('+')[0]))
        retval['return_code'] = 1
        return retval

    if bids_dir in opts.work_dir.parents:
        build_log.error(
            'The selected working directory is a subdirectory of the input BIDS folder. '
            'Please modify the output path.')
        retval['return_code'] = 1
        return retval

    # Set up some instrumental utilities
    run_uuid = '%s_%s' % (strftime('%Y%m%d-%H%M%S'), uuid.uuid4())
    retval['run_uuid'] = run_uuid
    if opts.participant_label:
        retval['participant_label'] = opts.participant_label
    else:
        from bids import BIDSLayout
        retval['participant_label'] = BIDSLayout(opts.bids_dir).get_subjects()

    # Load base plugin_settings from file if --use-plugin
    if opts.use_plugin is not None:
        with open(opts.use_plugin) as f:
            plugin_settings = json.load(f)
    else:
        # Defaults
        plugin_settings = {
            'plugin': 'MultiProc',
            'plugin_args': {
                'raise_insufficient': False,
                'maxtasksperchild': 1,
            }
        }

    # Resource management options
    # Note that we're making strong assumptions about valid plugin args
    # This may need to be revisited if people try to use batch plugins
    #nthreads = plugin_settings['plugin_args'].get('n_procs')
    # Permit overriding plugin config with specific CLI options
    #if nthreads is None or opts.nthreads is not None:
    #    nthreads = opts.nthreads
    #    if nthreads is None or nthreads < 1:
    #        nthreads = cpu_count()
    #    plugin_settings['plugin_args']['n_procs'] = nthreads

    #if opts.mem_mb:
    #    plugin_settings['plugin_args']['memory_gb'] = opts.mem_mb / 1024

    #omp_nthreads = opts.omp_nthreads
    #if omp_nthreads == 0:
    #    omp_nthreads = min(nthreads - 1 if nthreads > 1 else cpu_count(), 8)

    #if 1 < nthreads < omp_nthreads:
    #    build_log.warning(
    #        'Per-process threads (--omp-nthreads=%d) exceed total '
    #        'threads (--nthreads/--n_cpus=%d)', omp_nthreads, nthreads)
    retval['plugin_settings'] = plugin_settings

    # Set up directories
    log_dir = output_dir / 'logs'
    # Check and create output and working directories
    output_dir.mkdir(exist_ok=True, parents=True)
    log_dir.mkdir(exist_ok=True, parents=True)
    work_dir.mkdir(exist_ok=True, parents=True)

    # Nipype config (logs and execution)
    ncfg.update_config({
        'logging': {
            'log_directory': str(log_dir),
            'log_to_file': True
        },
        'execution': {
            'crashdump_dir': str(log_dir),
            'crashfile_format': 'txt',
            'get_linked_libs': False,
            #'stop_on_first_crash': opts.stop_on_first_crash,
        },
        'monitoring': {
            #'enabled': opts.resource_monitor,
            'sample_frequency': '0.5',
            'summary_append': True,
        }
    })

    #if opts.resource_monitor:
    #    ncfg.enable_resource_monitor()

    # Called with reports only
    #if opts.reports_only:
    #    build_log.log(25, 'Running --reports-only on participants %s', ', '.join(opts.participant_label))
    #    if opts.run_uuid is not None:
    #        run_uuid = opts.run_uuid
    #        retval['run_uuid'] = run_uuid
    #    retval['return_code'] = generate_reports(
    #        opts.participant_label, output_dir, work_dir, run_uuid,
    #        packagename='funcworks')
    #    return retval

    # Build main workflow
    build_log.log(25, INIT_MSG(version=__version__,
                               bids_dir=bids_dir,
                               participant_label=opts.participant_label,
                               uuid=run_uuid))

    retval['workflow'] = init_funcworks_wf(model_file=opts.model_file,
                                           bids_dir=opts.bids_dir,
                                           output_dir=opts.output_dir,
                                           work_dir=opts.work_dir,
                                           participants=retval['participant_label'],
                                           analysis_level=opts.analysis_level,
                                           smoothing=opts.smoothing,
                                           derivatives=opts.derivatives,
                                           run_uuid=run_uuid,
                                           use_rapidart=opts.use_rapidart)
    retval['return_code'] = 0

    logs_path = Path(output_dir) / 'logs'
    boilerplate = retval['workflow'].visit_desc()

    if boilerplate:
        citation_files = {
            ext: logs_path / ('CITATION.%s' % ext)
            for ext in ('bib', 'tex', 'md', 'html')
        }
        # To please git-annex users and also to guarantee consistency
        # among different renderings of the same file, first remove any
        # existing one
        for citation_file in citation_files.values():
            try:
                citation_file.unlink()
            except FileNotFoundError:
                pass

        citation_files['md'].write_text(boilerplate)
        build_log.log(25, 'Works derived from this FUNCWorks execution should '
                      'include the following boilerplate:\n\n%s', boilerplate)
    return retval

if __name__ == '__main__':
    main()
