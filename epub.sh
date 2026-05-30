#!/bin/zsh -e
# Check if the required argument is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <EPUB_FILE|ACSM_FILE>"
    exit 1
fi

# If an ACSM file is provided, convert it to EPUB first
INPUT_FILE="$1"
if [[ "$INPUT_FILE" == *.acsm ]]; then
    docker run -it -v "$(pwd)":/home/knock -v ~/.config/knock:/root/.config/knock --rm jeffrice/docker-knock "$INPUT_FILE"
    EPUB_FILE="${INPUT_FILE%.acsm}.epub"
else
    EPUB_FILE="$INPUT_FILE"
fi

# Define variables
OUTPUT_DIR="$(uuidgen)"
DOCKER_IMAGE="ghcr.io/p0n1/epub_to_audiobook:latest"
VOICE_NAME="en-US-AvaNeural"
JOBS=4

# Run the Docker container to convert EPUB to audiobook (generates MP3 files)
docker run -i -t --rm -v "$(pwd)":/app "$DOCKER_IMAGE" "$EPUB_FILE" "$OUTPUT_DIR" \
    --tts edge --voice_name "$VOICE_NAME" --no_prompt --worker_count "$JOBS" --save_cover --chapter_summary --summary_base_url "http://host.docker.internal:8080/v1"

# Fix MP3 durations by re-encoding with FFmpeg in parallel (in-place)
# Hardcoded to match original TTS output: 32 kbps CBR, mono, 24 kHz
parallel -j "$JOBS" 'ffmpeg -y -i "{}" -c:a libmp3lame -b:a 32k -ac 1 -ar 24000 "{}.tmp.mp3" && mv "{}.tmp.mp3" "{}"' ::: "$OUTPUT_DIR"/*.mp3

# Extract the base filename without extension
BASENAME=$(basename -- "$EPUB_FILE")
FILENAME="${BASENAME%.*}"

m4b-tool() {
  docker run -it --rm -u "$(id -u)":"$(id -g)" -v "$(pwd)":/mnt sandreas/m4b-tool:latest "$@"
}

# Merge the fixed MP3s into a single M4B audiobook
m4b-tool merge "$OUTPUT_DIR" -f --jobs="$JOBS" --output-file="${FILENAME}.m4b"

# Notify user of successful completion
echo "Audiobook creation completed successfully!"