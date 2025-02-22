#!/bin/bash

SOURCE_DIR="/run/media/$USER/USB-DISK/RECORD"
DEST_DIR="$HOME/AudioRecordings"
OBSIDIAN_DIR="$HOME/Obsidian/VoiceNotes"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p "$DEST_DIR"
mkdir -p "$OBSIDIAN_DIR"

convert_date_format() {
    local filename="$1"
    local year="${filename:1:4}"
    local day="${filename:5:2}"
    local month="${filename:7:2}"
    local hour="${filename:10:2}"
    local minute="${filename:12:2}"
    echo "${day}-${month}-${year}_${hour}-${minute}.md"
}

for file in "$SOURCE_DIR"/*.WAV; do
    if [ -f "$file" ]; then
        base_filename=$(basename "$file")
        dest_file="$DEST_DIR/$base_filename"
        if [ ! -f "$dest_file" ]; then
            cp "$file" "$dest_file"
            echo "Copied $file to $dest_file"
        else
            echo "File $dest_file already exists, skipping."
        fi
    fi
done

for wav_file in "$DEST_DIR"/*.WAV; do
    if [ -f "$wav_file" ]; then
        base_filename=$(basename "$wav_file")
        txt_filename=$(convert_date_format "$base_filename")
        obsidian_file="$OBSIDIAN_DIR/$txt_filename"

        if [ ! -f "$obsidian_file" ]; then
            whisper --language ru -f txt --model turbo --threads 8 -o /tmp "$wav_file" >"$obsidian_file"
            # sed -i 's/^\[.*?\]//' "$obsidian_file"
            echo "Converted $wav_file to $obsidian_file"
        else
            echo "File $obsidian_file already exists, skipping conversion."
        fi
    fi
done

echo "Processing complete."
