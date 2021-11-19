#!/usr/bin/env bash
if [ "$#" -lt 1 ]; then
    echo "usage: build_dataset.sh <source-directory> [<output-directory>]"
    exit -1
fi

# Echo all commands / exit on first error
set -x -e

#%% Setup
# Required: Classes to choose from each of the source datasets
ESC50_TARGETS="footsteps dog door_wood_knock laughing coughing cat"
MSSNSD_TARGETS="AirConditioner ShuttingDoor Babble Restaurant Cafe"
DEMAND_TARGETS="DWASHING DLIVING DKITCHEN"
YOUTUBE_TARGETS="glassbreak"

# Required: What class(es) to target
# NB: careful not to duplicate classes
TARGET_CLASSES="glassbreak"
UNKNOWN_CLASSES=""
NOISE_CLASSES="$MSSNSD_TARGETS $DEMAND_TARGETS $ESC50_TARGETS"

# Source data directories
SRCDIR=${1}
DEMAND_SRCDIR=$SRCDIR/DEMAND
MSSNSD_SRCDIR=$SRCDIR/MSSNSD
ESC50_SRCDIR=$SRCDIR/ESC50
YOUTUBE_SRCDIR=$SRCDIR/YOUTUBE

# Output directory
DSTDIR=${2:-.}

# Parameters used for splitting up the data into segments
PYTHON="python"
RMDIR="rm -rf"
SPLIT_SEGMENTS_ARGS='--segment-length=3-5 --silence-threshold=-60 --min-silence-duration=0.5'

# Parameters for splitting segments into train and test folds
SPLIT_FOLDS_ARGS='--balance --copy --test-ratio=0.2'

mkdir -p "$DSTDIR"

#%% Split data into segments
SRCDS="ESC50"
for class in $ESC50_TARGETS; do
    ODIR="${DSTDIR}/${SRCDS}_${class}"
    # Map class string label to class ID
    classID=$(python -c "import json; print(json.load(open("\""esc50-class-map.json"\""))["\"$class\""])")
    # Note: ESC50 file names have format: {FOLD}-{CLIP_ID}-{TAKE}-{TARGET}.wav
    $RMDIR "$ODIR"
    $PYTHON split_into_segments.py $SPLIT_SEGMENTS_ARGS "${ESC50_SRCDIR}"'/*-'"${classID}.wav" "$ODIR"
done

SRCDS="DEMAND"
for class in $DEMAND_TARGETS; do
    ODIR="${DSTDIR}/${SRCDS}_${class}"
    $RMDIR "$ODIR"
    $PYTHON split_into_segments.py $SPLIT_SEGMENTS_ARGS "${DEMAND_SRCDIR}/${class}.wav" "$ODIR"
done

SRCDS="MSSNSD"
for class in $MSSNSD_TARGETS; do
    ODIR="${DSTDIR}/${SRCDS}_${class}"
    $RMDIR $ODIR
    $PYTHON split_into_segments.py $SPLIT_SEGMENTS_ARGS "${MSSNSD_SRCDIR}/${class}_"'*.wav' "$ODIR"
done

SRCDS="YOUTUBE"
for class in $YOUTUBE_TARGETS; do
    ODIR="${DSTDIR}/${SRCDS}_${class}"
    $RMDIR $ODIR
    $PYTHON split_into_segments.py $SPLIT_SEGMENTS_ARGS "${YOUTUBE_SRCDIR}/${class}/"'*.wav' "$ODIR"
done

#%% Define how data sources are assigned to classes
set +x
NOISE_FILES=""
for trgt in $NOISE_CLASSES; do 
    NOISE_FILES+=$'\n\t'""\""${DSTDIR}/*_${trgt}/*.wav"\"", "
done

UNKNOWN_FILES=""
for trgt in $UNKNOWN_TARGETS; do 
    UNKNOWN_FILES+=$'\n\t'""\""${DSTDIR}/*_${trgt}/*.wav"\"", "
done

TARGETS=""
for class in $TARGET_CLASSES; do
    TARGETS+=""\""${class}"\"": ["\""${DSTDIR}/*_$class/*.wav"\""], "$'\n\t'
done
set -x

tee "${DSTDIR}/input.json" <<EOF
{
    ${TARGETS%,*},
    "noise": [${NOISE_FILES%,*}],
    "unknown": [${UNKNOWN_FILES%,*}]
}
EOF

#%% Split into train and test folds
$PYTHON split_into_train_test.py $SPLIT_FOLDS_ARGS "${DSTDIR}/input.json" "$DSTDIR"
echo "Dataset output to ${DSTDIR}/train ${DSTDIR}/test"

pushd "$DSTDIR"

#%% Find and remove duplicates
# cat train_list.txt test_list.txt | xargs -P4 -d'\n' md5sum > md5_list.txt
# awk '{print $1}' md5_list.txt | sort | uniq -d | while read line; do rm `grep "$line" md5_list.txt | awk '{print substr($2,2); }'`; done
# rm *_list.txt
# find train/ -maxdepth 1 -name '*.wav'  > train_list.txt
# find test/ -maxdepth 1 -name '*.wav' > test_list.txt


#%% Upload directly to Edge Impulse
# xargs -d'\n' < train_list.txt edge-impulse-uploader --category training
# xargs -d'\n' < test_list.txt edge-impulse-uploader --category testing

popd