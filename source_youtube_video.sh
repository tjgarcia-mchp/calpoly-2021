#!/usr/bin/env bash
if [ "$#" -lt 1 ]; then
    echo "usage: source_youtube_video.sh youtube-vid [<output-directory>]"
    exit -1
fi

set -ex

# Video ID
VID=$1

# Output directory
DSTDIR=${2:-.}; DSTDIR=${DSTDIR%/}

### Specify path to all the required executables 
# See http://ytdl-org.github.io/youtube-dl/download.html
YOUTUBEDL=youtube-dl

# See https://ffmpeg.org/download.html
FFMPEG=ffmpeg

# AAC decoder: http://faac.sourceforge.net/ (or https://opus-codec.org/downloads/ for opus format)
FAAD=faad

# SoX: https://sourceforge.net/projects/sox/files/latest/download
SOX=sox
###

mkdir -p "$DSTDIR"

# Download audio
if ! test -f ${DSTDIR}/${VID}.wav; then
    # Download in best AAC format
    $YOUTUBEDL -f140 -x https://www.youtube.com/watch?v=${VID} -o ${DSTDIR}/'%(id)s.%(ext)s'

    # Convert to proper AAC
    $FFMPEG -i ${DSTDIR}/${VID}.m4a -c copy ${DSTDIR}/${VID}.aac

    # Decode and convert to 16k Mono WAV
    $FAAD -f 2 -b 1 -w ${DSTDIR}/${VID}.aac | $SOX -GD -r44100 -b16 -esigned -c2 -traw - -r16k -b16 -esigned -c1 $DSTDIR/${VID}.wav
fi

# Split into 1-hour segments
NSECS=$($SOX --i -D ${DSTDIR}/$VID.wav | cut -f 1 -d .)
SEGLEN=3600
eval echo -n {0..$NSECS..$SEGLEN} | xargs -t -d' ' -P4 -i sh -c 't={}; '"$SOX ${VID}"'.wav '"${DSTDIR}/${VID}.pt"'$(expr ${t} / '"$SEGLEN).wav trim {} $SEGLEN"