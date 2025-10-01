import logging
import numpy as np
import pickle
from pathlib import Path

from population_analysis import helperFns as hf


def plot():
    """Plot summary figures for a session, at the end of pipeline processing.

    Our Geffen lab ephys pipeline will process and summarize data as a Python .pkl, then call scripts like this one to create summary plots.
    The goal is to standardize the .pkl with summary data across the lab so the pipeline code will do this part.

    But we might want different summary plots, depending on the experiment.
    We can add different plotting functions in the plotting_scripts/ directory of this repo.

    At the end of processing the pipeline will call one or more of these plotting_scripts.
    Which plotting scripts to call can be configured along with other
    [pipeline configuration](https://github.com/benjamin-heasly/geffenlab-ephys-pipeline/blob/master/pipeline-configurations.md)

    All of the plotting scripts should have the same "shape" so that the pipeline can call them interchangeably:

        - The plotting script should define a plot() function, like this one.
        - The plot() function should work without any arguments passed to it.
        - The plot() function should expect to run in the "analysis" subdir of the session being processed.
        - The pipeline will set this as the working directory before calling plot().
        - The plot() function should look for a file named "summary.pkl" in the current directory.
        - Everything the plot() funciton needs should be found within "summary.pkl".
        - The plot() function should write figures into the "figures" subdirectory of the current directory.

    The "summary.pkl" for the current session will contain the following items:

        - subject: str subject id
        - date: str session date MMDDYYYY
        - session_info: dict of additional session metadata, optional
        - trial_events: dataframe of trial behavioral events
        - spikes_df: dataframe of sorted spikes
        - cluster_info: dataframe of sorted cluster labels and quality metrics
        - stim_tensor: ND array of cluster and spike data, arranged around stimulus time
        - resp_tensor: ND array of cluster and spike data, arranged around response time
    """

    # Expect to run from the analysis subdirectory for the current session.
    # Expect "summary.pkl" in this directory.
    pkl_path = Path("summary.pkl")
    logging.info(f"Loading data: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        df_dict = pickle.load(f)

    # Write figures into a "figures" subdirectory.
    figures_path = Path("figures")
    figures_path.mkdir(exist_ok=True, parents=True)

    # Unpack data we need from the pickled dictionary.
    subject = df_dict["subject"]
    date = df_dict["date"]
    trial_events = df_dict["trial_events"]
    kept_clusters = df_dict["kept_clusters"]
    spikes_df = df_dict["spikes_df"]

    # Sort units according to d-prime.
    logging.info(f"Sorting units by d-prime.")
    unique_stims = np.unique(trial_events['stim'])
    probe_stims = unique_stims[unique_stims > 14.0]
    effect_df, pcnt_stim, pcnt_cat = hf.make_effect_df(
        kept_clusters,
        trial_events['stim_time'],
        spikes_df,
        trial_events,
        probe_stims=probe_stims
    )
    values = np.abs(effect_df['onset_categorical_d']).values
    ids = kept_clusters
    valid_mask = ~np.isnan(values)
    valid_ids = ids[valid_mask]
    valid_values = values[valid_mask]
    sorted_ids = valid_ids[np.argsort(valid_values)[::-1]]

    logging.info(f"Sorted units by d-prime: {sorted_ids}")

    # Generate session summary plots.
    logging.info(f"Creating complex_condition_plot(s)")
    title = f"{subject}-{date}"
    hf.batch_plot(
        title,
        sorted_ids,
        spikes_df,
        trial_events,
        plot_fn=hf.complex_condition_plot,
        save_dir=figures_path.as_posix()
    )

    logging.info(f"Saved complex_condition_plot(s)")
