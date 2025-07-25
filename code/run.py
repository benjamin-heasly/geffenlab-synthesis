import sys
from argparse import ArgumentParser, BooleanOptionalAction
from typing import Optional, Sequence
import logging
from pathlib import Path
import json
import pickle

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
    title: str,
    session_info: str,
    data_path: Path,
    analysis_path: Path,
    results_path: Path,
    interneuron_search: bool,
    probe_stims: list[float],
    params_py_pattern: str,
    cluster_info_pattern: str,
    spike_times_sec_adj_pattern: str,
    event_times_pattern: str,
    behavior_txt_pattern: str,
    behavior_mat_pattern: str,
):
    logging.info("Synthesizing neural and behavioral data.\n")

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
    pkl_path = Path(results_path, f"{title}.pkl")
    logging.info("Saving data to .pkl.\n")
    all_clusters = np.unique(spikes_df['cluster'])
    stim_edges = np.arange(-0.5, 1.0, 0.02)
    stim_tensor = hf.gen_tensor(stim_edges, all_clusters, trial_events['stim_time'], spikes_df)
    resp_edges = np.arange(-1.0, 1.0, 0.02)
    resp_tensor = hf.gen_tensor(resp_edges, all_clusters, trial_events['resp_time'], spikes_df)
    df_dict = {
        "session_info": info,
        "trial_events": trial_events,
        "spikes_df": spikes_df,
        "cluster_info": cluster_info,
        "stim_tensor": stim_tensor,
        "stim_edges": stim_edges,
        "resp_tensor": resp_tensor,
        "resp_edges": resp_edges,
    }
    with open(pkl_path, 'wb') as f:
        pickle.dump(df_dict, f)

    logging.info("Creating summary plots.\n")

    # Sort units according to d-prime.
    if probe_stims is None:
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
    figures_path = Path(results_path, "figures")
    figures_path.mkdir(parents=True, exist_ok=True)
    hf.batch_plot(
        title,
        sorted_ids,
        spikes_df,
        trial_events,
        plot_fn=hf.complex_condition_plot,
        save_dir=figures_path.as_posix()
    )

    logging.info("OK\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    set_up_logging()

    parser = ArgumentParser(description="Synthesize neural and behavioral data, produce summary plots.")

    parser.add_argument(
        "--data-path", "-d",
        type=str,
        help="Where to find and read input data files for a session. (default: %(default)s)",
        default="/data"
    )
    parser.add_argument(
        "--analysis-path", "-a",
        type=str,
        help="Where to find and read input analysis products for a session. (default: %(default)s)",
        default="/analysis"
    )
    parser.add_argument(
        "--results-path", "-r",
        type=str,
        help="Where to write output result files. (default: %(default)s)",
        default="/results"
    )
    parser.add_argument(
        "--title", "-t",
        type=str,
        help="Title to use for summary figures. (default: %(default)s)",
        default="Multiplot!"
    )
    parser.add_argument(
        "--session-info", "-s",
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
        "--probe-stims", "-p",
        type=float,
        nargs="*",
        help="List of trial stim values to treat as probe stims.  Default is to take any stim values > 14.0.",
        default=[]
    )
    parser.add_argument(
        "--params-py-pattern", "-P",
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

    cli_args = parser.parse_args(argv)
    data_path = Path(cli_args.data_path)
    analysis_path = Path(cli_args.analysis_path)
    results_path = Path(cli_args.results_path)
    try:
        run_main(
            cli_args.title,
            cli_args.session_info,
            data_path,
            analysis_path,
            results_path,
            cli_args.interneuron_search,
            cli_args.probe_stims,
            cli_args.params_py_pattern,
            cli_args.cluster_info_pattern,
            cli_args.spike_times_sec_adj_pattern,
            cli_args.event_times_pattern,
            cli_args.behavior_txt_pattern,
            cli_args.behavior_mat_pattern,
        )
    except:
        logging.error("Error synthesizing session data.", exc_info=True)
        return -1


if __name__ == "__main__":
    exit_code = main(sys.argv[1:])
    sys.exit(exit_code)
