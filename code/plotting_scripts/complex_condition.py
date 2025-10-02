import logging
import numpy as np
import pickle
from pathlib import Path

from population_analysis import helperFns as hf


def plot():
    """Plot complex_condition_plot figures for a session using the lab's population-analysis code."""

    # Expect to run from the analysis subdirectory for the current session, with a "*summary.pkl".
    summary_paths = list(Path(".").glob("*summary.pkl"))
    logging.info(f"Found summary pickle: {summary_paths}")
    pkl_path = summary_paths[0]
    logging.info(f"Loading summary pickle: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        summary_dict = pickle.load(f)

    # Write figures into a "figures" subdirectory.
    figures_path = Path("figures")
    figures_path.mkdir(exist_ok=True, parents=True)

    # Unpack data we need from the pickled dictionary.
    subject = summary_dict["subject"]
    date = summary_dict["date"]
    trial_events = summary_dict["trial_events"]
    kept_clusters = summary_dict["kept_clusters"]
    spikes_df = summary_dict["spikes_df"]

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

    # Save one or more plots into the figures subdir.
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

    logging.info(f"Saved complex_condition_plot(s) to {figures_path}")


if __name__ == "__main__":
    # For testing locally, enable console logging and call plot() with no args.
    import sys
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    plot()
