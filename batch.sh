#!/bin/bash

INPUT_DIR="/mnt/DATA/DOWNLOADS_RESCUE/VIDEOS/ANTOONOV/MP3"
OUTPUT_DIR="transcripts" 
WHISPER_SCRIPT="src/main.py"
CONFIG_FILE="src/config.yaml"
OUTPUT_FORMAT="txt"

mkdir -p "${OUTPUT_DIR:-$INPUT_DIR}"

find "$INPUT_DIR" -type f -iname "*.mp3" -o -iname "*.wav" -o -iname "*.mp4" -o -iname "*.avi" -o -iname "*.mkv" -o -iname "*.mov"  | while read -r file; do
    filename=$(basename "$file")
    base_name="${filename%.*}"
    
    output_file="${base_name}_transcript.${OUTPUT_FORMAT}"
    
    if [ -z "$OUTPUT_DIR" ]; then
        output_path="$INPUT_DIR/$output_file"
    else
        output_path="$OUTPUT_DIR/$output_file"
    fi
    
    if [ -f "$output_path" ]; then
        echo "Skipping $filename - transcript already exists"
        continue
    fi
    
    echo "Processing: $filename"
    
    python3 "$WHISPER_SCRIPT" \
        --input "$file" \
        --output "$output_path" \
        --format "$OUTPUT_FORMAT" \
	--use-cuda \
	--remove-timestamps \
	${CONFIG_FILE:+--config "$CONFIG_FILE"}
done

echo "Transcription complete!"
