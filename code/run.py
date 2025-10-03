import sys
from importlib import import_module
from argparse import ArgumentParser, BooleanOptionalAction
from typing import Optional, Sequence
import logging
from pathlib import Path
import json
import pickle
from contextlib import chdir

import numpy as np

from population_analysis import loadFns as lf
from population_analysis import helperFns as hf


def set_up_logging():
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def run_main(
    experimenter: str,
    subject: str,
    date: str,
    session_info: str,
    raw_data_path: Path,
    processed_data_path: Path,
    results_path: Path,
    interneuron_search: bool,
    params_py_pattern: str,
    cluster_info_pattern: str,
    spike_times_sec_adj_pattern: str,
    event_times_pattern: str,
    behavior_txt_pattern: str,
    behavior_mat_pattern: str,
    stim_edges: list[float],
    resp_edges: list[float],
    pickle_name: str,
    plotting_scripts: list[str]
):
    logging.info(f"Synthesizing neural and behavioral data for {experimenter} {subject} {date}.\n")

    # Combine data from a few pipeline steps into one "neuronal location".
    neuronal_path = lf.combine_neural_data(
        analysis_path,
        params_py_pattern,
        cluster_info_pattern,
        spike_times_sec_adj_pattern,
        event_times_pattern
    )

    # Load the lab dataframes from local files.
    trial_events, spikes_df, cluster_info, kept_clusters, nb_times = lf.gen_dataframe_local(
        data_path,
        neuronal_path,
        interneuron_search,
        behavior_txt_pattern,
        behavior_mat_pattern
    )

    # Load optional session info from JSON.
    info = {}
    if session_info is not None:
        session_info_path = Path(session_info)
        if session_info_path.exists():
            logging.info(f"Loading session info from {session_info_path}.")
            with open(session_info_path, 'r') as f:
                info = json.load(f)
        else:
            logging.info(f"Loading session info from string {session_info}.")
            info = json.loads(session_info)

    # Save the synthesized session data to .pkl.
    results_path.mkdir(parents=True, exist_ok=True)
    pkl_path = Path(results_path, f"{experimenter}_{subject}_{date}_{pickle_name}")
    logging.info(f"Saving summary data to {pkl_path}\n")
    all_clusters = np.unique(spikes_df['cluster'])
    stim_edges_array = np.arange(stim_edges[0], stim_edges[1], stim_edges[2])
    stim_tensor = hf.gen_tensor(stim_edges_array, all_clusters, trial_events['stim_time'], spikes_df)
    resp_edges_array = np.arange(resp_edges[0], resp_edges[1], resp_edges[2])
    resp_tensor = hf.gen_tensor(resp_edges_array, all_clusters, trial_events['resp_time'], spikes_df)
    df_dict = {
        "experimenter": experimenter,
        "subject": subject,
        "date": date,
        "session_info": info,
        "trial_events": trial_events,
        "spikes_df": spikes_df,
        "cluster_info": cluster_info,
        "kept_clusters": kept_clusters,
        "nb_times": nb_times,
        "stim_tensor": stim_tensor,
        "stim_edges": stim_edges,
        "resp_tensor": resp_tensor,
        "resp_edges": resp_edges,
    }
    with open(pkl_path, 'wb') as f:
        pickle.dump(df_dict, f)

    logging.info("Creating summary plots.\n")
    for plotting_script in plotting_scripts:
        try:
            # Load the plotting script by name and get its plot() function.
            script_name = plotting_script.strip()
            logging.info(f"Plotting {script_name}")
            module_spec = f"plotting_scripts.{script_name}"
            module = import_module(module_spec)
            plot_function = getattr(module, 'plot')

            # Run the plot() function from the results dir that has summary.pkl.
            with chdir(results_path):
                logging.info(f"Running from {results_path}")
                plot_function()

        except Exception:
            logging.error(f"Error running plotting script {plotting_script}", exc_info=True)

    logging.info("OK\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    set_up_logging()

    parser = ArgumentParser(description="Synthesize neural and behavioral data, produce summary plots.")

    parser.add_argument(
        "--raw-data-path", "-D",
        type=str,
        help="Where to find and read raw data files for the session. (default: %(default)s)",
        default="/raw_data"
    )
    parser.add_argument(
        "--processed-data-path", "-P",
        type=str,
        help="Where to find and read processed data files for the session. (default: %(default)s)",
        default="/processed_data"
    )
    parser.add_argument(
        "--results-path", "-R",
        type=str,
        help="Where to write output results for the session. (default: %(default)s)",
        default="/results"
    )
    parser.add_argument(
        "--experimenter", "-e",
        type=str,
        help="Experimenter initials for the session being processed. (default: %(default)s)",
        default="BH"
    )
    parser.add_argument(
        "--subject", "-s",
        type=str,
        help="Subject ID for the session being processed. (default: %(default)s)",
        default="AS20-minimal2"
    )
    parser.add_argument(
        "--date", "-d",
        type=str,
        help="MMDDYYY date for the session being processed. (default: %(default)s)",
        default="03112025"
    )
    parser.add_argument(
        "--session-info", "-i",
        type=str,
        help="JSON string or file name with top-level session info to include. (default: %(default)s)",
        default=None
    )
    parser.add_argument(
        "--interneuron-search", "-I",
        action=BooleanOptionalAction,
        help="True or False, whether to analyze waveforms and identify interneurons. (default: %(default)s)",
        default=True
    )
    parser.add_argument(
        "--params-py-pattern", "-p",
        type=str,
        help="Glob pattern to locate Phy params.py within the session analysis subdir: ANALYSIS_ROOT/SUBJECT_ID/SESSION_DATE. (default: %(default)s)",
        default="exported/phy/*/params.py"
    )
    parser.add_argument(
        "--cluster-info-pattern", "-C",
        type=str,
        help="Glob pattern to locate Phy cluster_info.csv within the session analysis subdir: ANALYSIS_ROOT/SUBJECT_ID/SESSION_DATE. (default: %(default)s)",
        default="curated/*/cluster_info.tsv"
    )
    parser.add_argument(
        "--spike-times-sec-adj-pattern", "-S",
        type=str,
        help="Glob pattern to locate TPrime spike_times_sec_adj.npy within the session analysis subdir: ANALYSIS_ROOT/SUBJECT_ID/SESSION_DATE. (default: %(default)s)",
        default="exported/tprime/*/spike_times_sec_adj.npy"
    )
    parser.add_argument(
        "--event-times-pattern", "-E",
        type=str,
        help="Glob pattern to locate a trial events times text file within the session analysis subdir: ANALYSIS_ROOT/SUBJECT_ID/SESSION_DATE. (default: %(default)s)",
        default="exported/tprime/*/*nidq.xd_8_3_0.txt"
    )
    parser.add_argument(
        "--behavior-txt-pattern", "-T",
        type=str,
        help="Glob pattern to locate a behavior text file session data subdir: DATA_ROOT/SUBJECT_ID/SESSION_DATE. (default: %(default)s)",
        default="behavior/*.txt"
    )
    parser.add_argument(
        "--behavior-mat-pattern", "-M",
        type=str,
        help="Glob pattern to locate a behavior mat-file session data subdir: DATA_ROOT/SUBJECT_ID/SESSION_DATE. (default: %(default)s)",
        default="behavior/*.mat"
    )
    parser.add_argument(
        "--stim-edges",
        type=float,
        nargs="+",
        help="List of bin edge [low, high, step] for creating stim_tensor. (default: %(default)s)",
        default=[-0.5, 1.0, 0.02]
    )
    parser.add_argument(
        "--resp-edges",
        type=float,
        nargs="+",
        help="List of bin edge [low, high, step] for creating resp_tensor. (default: %(default)s)",
        default=[-1.0, 1.0, 0.02]
    )
    parser.add_argument(
        "--pickle-name",
        type=str,
        help="File name for the summary .pkl file to create. (default: %(default)s)",
        default="summary.pkl"
    )
    parser.add_argument(
        "--plotting_scripts",
        type=str,
        nargs="+",
        help="Names of plotting scripts to run at the end (found in plotting_scripts/ subdir, without .py at the end). (default: %(default)s)",
        default=["complex_condition demo"]
    )

    cli_args = parser.parse_args(argv)
    raw_data_path = Path(cli_args.raw_data_path)
    processed_data_path = Path(cli_args.processed_data_path)
    results_path = Path(cli_args.results_path)
    try:
        run_main(
            cli_args.experimenter,
            cli_args.subject,
            cli_args.date,
            cli_args.session_info,
            raw_data_path,
            processed_data_path,
            results_path,
            cli_args.interneuron_search,
            cli_args.params_py_pattern,
            cli_args.cluster_info_pattern,
            cli_args.spike_times_sec_adj_pattern,
            cli_args.event_times_pattern,
            cli_args.behavior_txt_pattern,
            cli_args.behavior_mat_pattern,
            cli_args.stim_edges,
            cli_args.resp_edges,
            cli_args.pickle_name,
            cli_args.plotting_scripts
        )
    except:
        logging.error("Error synthesizing session data.", exc_info=True)
        return -1


if __name__ == "__main__":
    exit_code = main(sys.argv[1:])
    sys.exit(exit_code)
