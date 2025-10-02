import logging
import pickle
from pathlib import Path

from matplotlib import pyplot as plt


def plot():
    """Demonstrate how to write a plotting script that fits within pipeline runs.

    You can copy and modify this script to create your own plotting scripts.

    For more about how to relase a plotting script and configure it into pipeline runs, see:
        https://github.com/benjamin-heasly/geffenlab-ephys-pipeline/blob/master/summary-plotting-scripts.md

    Plotting scripts all must have the same overall shape.  They must:

        - Have a plot() function like this one, that works with no arguments.
        - Expect a session data pickle in the current directory, with a name ending like "summary.pkl".
        - Save figures into a subdirectory called "figures/".

    The "summary.pkl" for the current session will contain a dictionary with items like:

        - "subject": str subject id
        - "date": str session date MMDDYYYY
        - "trial_events": dataframe of trial behavioral events
        - "spikes_df": dataframe of sorted spikes
        - "cluster_info": dataframe of sorted cluster labels and quality metrics
        - "stim_tensor": ND array of cluster and spike data, arranged around stimulus time
        - "resp_tensor": ND array of cluster and spike data, arranged around response time
    """

    # Expect to run from the analysis subdirectory for the current session.
    # Find and load a "*summary.pkl" in this directory.
    summary_paths = list(Path(".").glob("*summary.pkl"))
    logging.info(f"Found summary pickle: {summary_paths}")
    pkl_path = summary_paths[0]
    logging.info(f"Loading summary pickle: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        summary_dict = pickle.load(f)

    # Write figures into a "/" subdirectory.
    figures_path = Path("figures")
    figures_path.mkdir(exist_ok=True, parents=True)

    # Unpack data we need from the summary dictionary.
    subject = summary_dict["subject"]
    date = summary_dict["date"]
    trial_events = summary_dict["trial_events"]
    spikes_df = summary_dict["spikes_df"]

    # Choose an arbitrary time range to detail.
    detail_seconds = 10

    # Plot with matplotlib.
    fig, ax = plt.subplots(nrows=2, figsize=(15, 10))

    # Plot stim events with time and stim value.
    stim_selector = trial_events['stim_time'] <= detail_seconds
    ax[0].scatter(trial_events['stim_time'][stim_selector], trial_events['stim'][stim_selector])
    ax[0].set_title(f"{subject}_{date}_demo (first {detail_seconds}s)")
    ax[0].set_ylabel("stim")
    unique_stims = list(set(trial_events['stim'][stim_selector]))
    unique_stims.sort()
    ax[0].set_yticks(unique_stims)
    ax[0].grid()
    ax[0].set_xlim((0, detail_seconds))

    # Plot spike events with time and cluster id.
    spike_selector = spikes_df['time'] < detail_seconds
    ax[1].scatter(spikes_df['time'][spike_selector], spikes_df['cluster'][spike_selector], marker='|')
    ax[1].set_ylabel("cluster")
    ax[1].set_xlabel(f"aligned time (s)")
    unique_clusters = list(set(spikes_df['cluster'][spike_selector]))
    unique_clusters.sort()
    ax[1].set_yticks(unique_clusters)
    ax[1].grid()
    ax[1].set_xlim((0, detail_seconds))

    # Save the figure as a .png.
    png_path = Path(figures_path, f"{subject}-{date}_demo.png")
    fig.savefig(png_path)
    logging.info(f"Saved plot to {png_path}")


if __name__ == "__main__":
    # For testing locally, enable console logging and call plot() with no args.
    import sys
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    plot()
