#!/bin/bash
if (( "$#" < 2 )); then 
    echo "Pass YAML config file and script to test."
    exit 1
fi

export CONFIG_FILE="$1"
export PBS_ARRAY_INDEX=0
eval "./$2 ${@:3}" 
