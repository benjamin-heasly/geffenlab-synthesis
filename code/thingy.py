# conda activate minimal-dataset

from pathlib import Path

import loadFns as lf
import helperFns as hf
import numpy as np


data_root = Path("/home/ninjaben/codin/geffen-lab-data/data")
analysis_root = Path("/home/ninjaben/codin/geffen-lab-data/analysis")

# Combine data from a few pipeline steps into one "neuronal location".
ID = 'AS20-minimal'
session_date = '03112025'
neuronal_path = lf.combine_neural_data(analysis_root, ID, session_date)

# Load the lab standard dataframes from local files.
trial_events, spikes_df, cluster_info, kept_clusters, nb_times = lf.gen_dataframe_local(
    data_root, analysis_root, neuronal_path, ID, session_date)

# Sort trials according to d-prime.
unique_stims = np.unique(trial_events['stim'])
probe_stims = unique_stims[unique_stims > 14.0]
effect_df, pcnt_stim, pcnt_cat = hf.make_effect_df(kept_clusters,
                                                   trial_events['stim_time'],
                                                   spikes_df,
                                                   trial_events,
                                                   probe_stims=probe_stims)
values = np.abs(effect_df['onset_categorical_d']).values
ids = kept_clusters
valid_mask = ~np.isnan(values)
valid_ids = ids[valid_mask]
valid_values = values[valid_mask]
sorted_indices = np.argsort(valid_values)
sorted_ids_asc = valid_ids[sorted_indices]
sorted_values = valid_values[sorted_indices]
sorted_ids = valid_ids[np.argsort(valid_values)[::-1]]

print(sorted_ids)

# Generate the lab standard summary plots.
plots_path = Path(analysis_root, ID, session_date, "multiplot")
plots_path.mkdir(parents=True, exist_ok=True)
hf.batch_plot(ID + '_' + session_date,
              sorted_ids,
              spikes_df,
              trial_events,
              plot_fn = hf.complex_condition_plot,
              save_dir = plots_path.as_posix())
