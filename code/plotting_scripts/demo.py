import logging
import pickle
from pathlib import Path


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
    trial_events = df_dict["trial_events"]
    spikes_df = df_dict["spikes_df"]

    logging.info(f"Saving demo plots")


    logging.info(f"Saved demo plots")


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    plot()
