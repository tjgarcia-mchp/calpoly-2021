#!/usr/bin/env bash
#%% Setup
if [ "$#" -lt 1 ]; then
    echo "usage: build_dataset.sh [-f] <source-directory> [<output-directory>]"
    exit -1
fi

if [ $1 = "-f" ]; then
    OVERWRITE=1
    shift
else
    OVERWRITE=0
fi

# Echo all commands / exit on first error
set -x -e

# Source data directories
SRCDIR=${1}
DEMAND_SRCDIR=$SRCDIR/DEMAND
MSSNSD_SRCDIR=$SRCDIR/MSSNSD
ESC50_SRCDIR=$SRCDIR/ESC50
YOUTUBE_SRCDIR=$SRCDIR/YOUTUBE
KEYWORDS_SRCDIR=$SRCDIR/keywords

# Output directory
DSTDIR=${2:-.}

PYTHON="python"
RMDIR="rm -rf"

# Parameters used for splitting up the data into segments
SPLIT_SEGMENTS_ARGS='--segment-length=3-5 --silence-threshold=-60 --min-silence-duration=0.5'

# Parameters for splitting segments into train and test folds
SPLIT_FOLDS_ARGS='--balance-classes --test-ratio=0.2 --copy'

#%% Data sources
# Required: Classes to choose from each of the source datasets
ESC50_TARGETS=$(tr '[:space:]' ' ' <<__EOF__
footsteps
dog
door_wood_knock
laughing
coughing
cat
babycry
__EOF__
)

MSSNSD_TARGETS=$(tr '[:space:]' ' ' <<__EOF__
AirConditioner
ShuttingDoor
Babble
Restaurant
Cafe
__EOF__
)

DEMAND_TARGETS=$(tr '[:space:]' ' ' <<__EOF__
DWASHING
DLIVING
DKITCHEN
__EOF__
)

YOUTUBE_TARGETS=$(tr '[:space:]' ' ' <<__EOF__
glassbreak
babycry
__EOF__
)

KEYWORDS_TARGETS=$(tr '[:space:]' ' ' <<__EOF__
__EOF__
)

# Required: What class(es) to target
# NB: careful not to duplicate classes
TARGET_CLASSES="glassbreak babycry"

UNKNOWN_CLASSES=""

NOISE_CLASSES="$MSSNSD_TARGETS $DEMAND_TARGETS"

#%% Generate segments
mkdir -p "$DSTDIR"

SRCDS="ESC50"
for class in $ESC50_TARGETS; do
    ODIR="${DSTDIR}/${SRCDS}_${class}"
    if [[ -d "$ODIR" && $OVERWRITE -eq 0 ]]; then
        continue
    fi
    $RMDIR "$ODIR"

    # Map class string label to class ID
    classID=$(python -c "import json; print(json.load(open("\""esc50-class-map.json"\""))["\"$class\""])")    
    $PYTHON split_into_segments.py $SPLIT_SEGMENTS_ARGS "${ESC50_SRCDIR}"'/*-'"${classID}.wav" "$ODIR"
done

SRCDS="DEMAND"
for class in $DEMAND_TARGETS; do
    ODIR="${DSTDIR}/${SRCDS}_${class}"
    if [[ -d "$ODIR" && $OVERWRITE -eq 0 ]]; then
        continue
    fi    
    $RMDIR "$ODIR"
    $PYTHON split_into_segments.py $SPLIT_SEGMENTS_ARGS "${DEMAND_SRCDIR}/${class}.wav" "$ODIR"
done

SRCDS="MSSNSD"
for class in $MSSNSD_TARGETS; do
    ODIR="${DSTDIR}/${SRCDS}_${class}"
    if [[ -d "$ODIR" && $OVERWRITE -eq 0 ]]; then
        continue
    fi    
    $RMDIR "$ODIR"
    $PYTHON split_into_segments.py $SPLIT_SEGMENTS_ARGS "${MSSNSD_SRCDIR}/${class}_"'*.wav' "$ODIR"
done

SRCDS="YOUTUBE"
for class in $YOUTUBE_TARGETS; do
    ODIR="${DSTDIR}/${SRCDS}_${class}"
    if [[ -d "$ODIR" && $OVERWRITE -eq 0 ]]; then
        continue
    fi    
    $RMDIR "$ODIR"
    $PYTHON split_into_segments.py $SPLIT_SEGMENTS_ARGS "${YOUTUBE_SRCDIR}/${class}/"'*.wav' "$ODIR"
done

SRCDS="keywords"
for class in $KEYWORDS_TARGETS; do
    ODIR="${DSTDIR}/${SRCDS}_${class}"
    if [[ -d "$ODIR" && $OVERWRITE -eq 0 ]]; then
        continue
    fi
    $RMDIR "$ODIR"

    $PYTHON split_into_segments.py --segment-length=1- --silence-threshold=0 "${KEYWORDS_SRCDIR}/${class}/"'*.wav' "$ODIR"
done

#%% Define how data sources are assigned to classes
# Shh
set +x

NOISE_FILES=""
for trgt in $NOISE_CLASSES; do 
    NOISE_FILES+=$'\n\t'""\""${DSTDIR}/*_${trgt}/*.wav"\"", "
done

UNKNOWN_FILES=""
for trgt in $UNKNOWN_CLASSES; do 
    UNKNOWN_FILES+=$'\n\t'""\""${DSTDIR}/*_${trgt}/*.wav"\"", "
done

TARGETS=""
for class in $TARGET_CLASSES; do
    TARGETS+=""\""${class}"\"": ["\""${DSTDIR}/*_$class/*.wav"\""], "$'\n\t'
done

tee "${DSTDIR}/input.json" <<__EOF__
{
    ${TARGETS%,*},
    "noise": [${NOISE_FILES%,*}],
    "unknown": [${UNKNOWN_FILES%,*}]
}
__EOF__

set -x

#%% Split into train and test folds
$PYTHON split_into_train_test.py $SPLIT_FOLDS_ARGS "${DSTDIR}/input.json" "$DSTDIR"

pushd "$DSTDIR" &> /dev/null

#%% Find and remove duplicates
# cat train_list.txt test_list.txt | xargs -P4 -d'\n' md5sum > md5_list.txt
# awk '{print $1}' md5_list.txt | sort | uniq -d | while read line; do rm `grep "$line" md5_list.txt | awk '{print substr($2,2); }'`; done
# rm *_list.txt
# find train/ -maxdepth 1 -name '*.wav'  > train_list.txt
# find test/ -maxdepth 1 -name '*.wav' > test_list.txt

#%% Upload directly to Edge Impulse
# xargs -d'\n' < train_list.txt edge-impulse-uploader --category training
# xargs -d'\n' < test_list.txt edge-impulse-uploader --category testing

popd &> /dev/null