#!/bin/sh

set -e

docker build -f environment/Dockerfile -t geffenlab/synthesis:local .
