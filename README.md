# geffenlab-synthesis

Synthesize neural and behavioral data into dataframes and summary plots.

This container incorporates code from the lab's non-public [population-analysis](https://github.com/jcollina/population-analysis.git) repo.
To simplify the Docker build, we obtain this repo as a Git submodule in `code/population-analysis/`.
To get the current code from this repo and the submodule, you can `pull` with submodules.

```
git pull --recurse-submodules
```

# Building Docker image versions

This repo is configured to automatically build and publish a new Docker image version, each time a [repo tag](https://git-scm.com/book/en/v2/Git-Basics-Tagging) is pushed to GitHub.

## Published versions

The published images are located in the GitHub Container Registry as [geffenlab-synthesis](https://github.com/benjamin-heasly/geffenlab-synthesis/pkgs/container/geffenlab-synthesis).  You can find the latest public version at this page.

You can access published images using their full names.  For version `v0.0.4` the full name would be `ghcr.io/benjamin-heasly/geffenlab-synthesis:v0.0.4`.  You can use this name in [Nexflow pipeline configuration](https://github.com/benjamin-heasly/geffenlab-ephys-pipeline/blob/master/pipeline/main.nf#L128) and with Docker commands like:

```
docker pull ghcr.io/benjamin-heasly/geffenlab-synthesis:v0.0.4
```

## Releasing new versions

Here's a workflow for building and realeasing a new Docker image version.

First, make changes to the code in this repo, and `push` the changes to GitHub.

```
# Edit code
git commit -a -m "Now with lasers!"
git push
```

next, Create a new repository [tag](https://git-scm.com/book/en/v2/Git-Basics-Tagging), which marks the most recent commit as important, giving it a unique name and description.

```
# Review existing tags and choose the next version number to use.
git pull --tags
git tag -l

# Create the tag for the next version
git tag -a v0.0.5 -m "Now with lasers!"
git push --tags
```

GitHub should automatically kick off a build and publish workflow for the new tag.
You can follow the workflow progress at the repo's [Actions](https://github.com/benjamin-heasly/geffenlab-synthesis/actions) page.

You can see the workflow code in [build-tag.yml](./.github/workflows/build-tag.yml).


## Access to `population-analysis` repo

The release automation relies on a [GitHhub secret](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-secrets) named `POPULATION_ANALYSIS_READ`.  The value of this secret must be a [Personal Access Token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) with permission to read from the [population-analysis](https://github.com/jcollina/population-analysis.git) repo.
