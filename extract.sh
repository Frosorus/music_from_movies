#!/bin/bash
# Extract subtitles from each MKV file in the given directory

DIR="/mnt/e/Videos/Films"
mkdir -p "$DIR/subtitles"
# Get all the MKV files in this dir and its subdirs
find "$DIR" -type f -name '*.mkv' | while read filename
do
  # Find out which tracks contain the subtitles
  mkvmerge -i "$filename" | grep 'subtitles' | while read subline
  do
    # Grep the number of the subtitle track
    tracknumber=`echo $subline | egrep -o "[0-9]{1,2}" | head -1`

    # Get base name for subtitle
    subtitlename=${filename%.*}

    # Extract the track to a .tmp file
    `mkvextract tracks "$filename" $tracknumber:"$subtitlename.srt.tmp" > /dev/null 2>&1`
    `chmod g+rw "$subtitlename.srt.tmp"`

    # Not our desired language: add a number to the filename and keep anyway, just in case
    `mv "$subtitlename.srt.tmp" "$DIR/subtitles/$(basename "$subtitlename").$tracknumber.srt" > /dev/null 2>&1`
    echo "Done"
  done
done