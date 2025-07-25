# geffenlab-synthesis
Synthesize neural and behavioral data into dataframes and summary plots.

This container incorporates code from the lab's non-private population-analysis repo:

https://github.com/jcollina/population-analysis.git

To simplify the Docker build, we obtain this repo as a git submodule in code/population-analysis.
To get all the latest code, you can pull with submodules.

```
git pull --recurse-submodules
```

Local testing:

```
DATA_ROOT=/home/ninjaben/codin/geffen-lab-data/data
ANALYSIS_ROOT=/home/ninjaben/codin/geffen-lab-data/analysis

SUBJECT=AS20-minimal2
DATE=03112025

DATA_PATH="$DATA_ROOT/$SUBJECT/$DATE"
ANALYSIS_PATH="$ANALYSIS_ROOT/$SUBJECT/$DATE"

docker run -ti --rm -u $(id -u):$(id -g) -v $DATA_PATH:$DATA_PATH -v $ANALYSIS_PATH:$ANALYSIS_PATH geffenlab/synthesis:local /opt/code/conda_run python /opt/code/run.py --data-path=$DATA_PATH --analysis-path=$ANALYSIS_PATH --results-path=$ANALYSIS_PATH/synthesis
```
