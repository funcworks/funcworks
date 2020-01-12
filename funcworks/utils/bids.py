import os
import json
from bids.layout import BIDSLayout

def collect_derivates(bids_dir, derivatives_dir, participant_label, task, bids_validate=True):
    if isinstance(bids_dir, BIDSLayout):
        layout = bids_dir
    else:
        layout = BIDSLayout(str(bids_dir), validate=bids_validate)

    queries = {
        'confounds': {'datatype': 'func', 'suffix':'regressors', 'desc':'confounds', 'extension':'tsv'},
        'bold': {'datatype': 'func', 'suffix': 'bold', 'desc':'preproc'},
        'events': {'datatype':'func', 'suffix':'events', 'extension':'tsv'}}
    
        
    if task:
        for query in queries:
            queries[query]['task'] = task
        
    subj_data = {
        dtype: sorted(layout.get(return_type='file', subject=participant_label,
                                 extension=['nii', 'nii.gz'], **query))
        for dtype, query in queries.items()}

    return subj_data, layout