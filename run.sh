#!/bin/bash

train() {
  mpirun ${MPIOPTS} \
  python3 tools/train.py \
    tools/train.py ${EXTRA_ARGS}
}


test() {
  mpirun ${MPIOPTS} \
  python3 tools/train.py \
    tools/test.py ${EXTRA_ARGS}
}

############################ Main #############################
GPUS=`nvidia-smi -L | wc -l`
MASTER_PORT=9000
INSTALL_DEPS=false

while [[ $# -gt 0 ]]
do

key="$1"
case $key in
    -h|--help)
    echo "Usage: $0 [run_options]"
    echo "Options:"
    echo "  -g|--gpus <8> - number of gpus to be used"
    echo "  -n|--nodes <1> - number of nodes"
    echo "  -t|--job-type <train> - job type (train|test)"
    exit 1
    ;;
    -g|--gpus)
    GPUS=$2
    shift
    ;;
    -n|--nodes)
    NODES=$2
    shift
    ;;
    -t|--job-type)
    JOB_TYPE=$2
    shift
    ;;
    *)
    EXTRA_ARGS="$EXTRA_ARGS $1"
    ;;
esac
shift
done

echo "job type: ${JOB_TYPE}"
GPN=$((GPUS/NODES))
MPIOPTS="-np ${GPUS} --map-by ppr:${GPN}:node"

case $JOB_TYPE in
    train)
    train
    ;;
    test)
    test
    ;;
    *)
    echo "unknown job type"
    ;;
esac