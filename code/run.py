import sys
from argparse import ArgumentParser
from typing import Optional, Sequence, Any
import logging
from pathlib import Path

from export_to_phy import export_phy
from create_cluster_info import create_cluster_info


def set_up_logging():
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def capsule_main(
    data_path: Path,
    results_path: Path,
    postprocessed_pattern: str,
    curated_pattern: str,
    compute_pc_features: bool,
    copy_binary: bool,
    n_jobs: int
):

    logging.info("Thingy.\n")
# trial_events, spikes_df, cluster_info, kept_clusters, nb_times = lf.gen_dataframe_local(
#     data_root, analysis_root, neuronal_path, ID, session_date)

# # Sort trials according to d-prime.
# unique_stims = np.unique(trial_events['stim'])
# probe_stims = unique_stims[unique_stims > 14.0]
# effect_df, pcnt_stim, pcnt_cat = hf.make_effect_df(kept_clusters,
#                                                    trial_events['stim_time'],
#                                                    spikes_df,
#                                                    trial_events,
#                                                    probe_stims=probe_stims)
# values = np.abs(effect_df['onset_categorical_d']).values
# ids = kept_clusters
# valid_mask = ~np.isnan(values)
# valid_ids = ids[valid_mask]
# valid_values = values[valid_mask]
# sorted_indices = np.argsort(valid_values)
# sorted_ids_asc = valid_ids[sorted_indices]
# sorted_values = valid_values[sorted_indices]
# sorted_ids = valid_ids[np.argsort(valid_values)[::-1]]

# print(sorted_ids)

# # Generate the lab standard summary plots.
# plots_path = Path(analysis_root, ID, session_date, "multiplot")
# plots_path.mkdir(parents=True, exist_ok=True)
# hf.batch_plot(ID + '_' + session_date,
#               sorted_ids,
#               spikes_df,
#               trial_events,
#               plot_fn = hf.complex_condition_plot,
#               save_dir = plots_path.as_posix())

    logging.info("OK\n")


def truthy_str(str_value: str) -> bool:
    """Parse a string argument value into a boolean value.

    Using bool as an argparse type doesn't do what we want.
    It gives bool('True') == True, but also bool('False') == True.
    This is becasue 'False' is a non-empty str.
    Likewise for bool('0'), bool('no'), etc.

    BooleanOptionalAction is a nicer, more idiomatic way to handle boolean args.
        https://docs.python.org/3/library/argparse.html#argparse.BooleanOptionalAction
    This sets up mutually exclusive flag arguments like --option vs --no-option.

    But as of writing, Code Ocean App Panels don't support flags like these.
    They only support arguments with explicit values like "--option value".
    So we'll use this function to parse the value into a bool.
    We can set up the App Panel to only pass in values we know how to parse, like "true" or "yes".
    """

    truthy_values = {'true', 't', 'yes', 'y', '1'}
    if str_value.lower() in truthy_values:
        return True
    else:
        return False


def main(argv: Optional[Sequence[str]] = None) -> int:
    set_up_logging()

    parser = ArgumentParser(description="Export ecephys sorting resluts to Phy.")

    parser.add_argument(
        "--data-root", "-d",
        type=str,
        help="Where to find and read input data files. (default: %(default)s)",
        default="/data"
    )
    parser.add_argument(
        "--analysis-root", "-a",
        type=str,
        help="Where to find and read input analysis products. (default: %(default)s)",
        default="/analysis"
    )
    parser.add_argument(
        "--results-root", "-r",
        type=str,
        help="Where to write output result files. (default: %(default)s)",
        default="/results"
    )
    parser.add_argument(
        "--subject-id", "-i",
        type=str,
        help="Mouse/subject id, for example 'AS20'"
    )
    parser.add_argument(
        "--session-date", "-s",
        type=str,
        help="Sessio date, for example '03112025'"
    )
    parser.add_argument(
        "--interneuron-search", "-I",
        type=truthy_str,
        help="True or False, whether to search for interneurons. (default: %(default)s)",
        default=False
    )
    parser.add_argument(
        "--probe-stims", "p",
        type=float,
        nargs="*",
        help="List of trial stim values to treat as probe stims.  Default is to take the top 6 out of 12 (assumed) stim values.",
        default = []
    )
    parser.add_argument(
        "--params-py-pattern", "-P",
        type=str,
        help="Glob pattern to locate Phy params.py within the session analysis subdir: ANALYSIS_ROOT/SUBJECT_ID/SESSION_DATE. (default: %(default)s)",
        default="exported/phy/**/params.py"
    )
    parser.add_argument(
        "--cluster-info-pattern", "-C",
        type=str,
        help="Glob pattern to locate Phy cluster_info.csv within the session analysis subdir: ANALYSIS_ROOT/SUBJECT_ID/SESSION_DATE. (default: %(default)s)",
        default="curated/**/cluster_info.tsv"
    )
    parser.add_argument(
        "--spike-times-sec-adj-pattern", "-S",
        type=str,
        help="Glob pattern to locate TPrime spike_times_sec_adj.npy within the session analysis subdir: ANALYSIS_ROOT/SUBJECT_ID/SESSION_DATE. (default: %(default)s)",
        default="exported/tprime/**/spike_times_sec_adj.npy"
    )
    parser.add_argument(
        "--event-times-pattern", "-E",
        type=str,
        help="Glob pattern to locate a trial events times text file within the session analysis subdir: ANALYSIS_ROOT/SUBJECT_ID/SESSION_DATE. (default: %(default)s)",
        default="exported/tprime/**/*nidq.xd_8_3_0.txt"
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
    data_path = Path(cli_args.data_root)
    analysis_path = Path(cli_args.analysis_root)
    results_path = Path(cli_args.results_root)
    try:
        capsule_main(
            data_path=data_path,
            results_path=results_path,
            postprocessed_pattern=cli_args.postprocessed_pattern,
            curated_pattern=cli_args.curated_pattern,
            compute_pc_features=cli_args.compute_pc_features,
            copy_binary=cli_args.copy_binary,
            n_jobs=cli_args.n_jobs
        )
    except:
        logging.error("Error synthesizing session data.", exc_info=True)
        return -1


if __name__ == "__main__":
    exit_code = main(sys.argv[1:])
    sys.exit(exit_code)
