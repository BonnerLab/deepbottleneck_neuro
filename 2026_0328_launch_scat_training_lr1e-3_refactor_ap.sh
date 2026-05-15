NOTEBOOK_DIR="/home/apassi1/deepbottleneck/notebooks"

cd "$NOTEBOOK_DIR"

shopt -s nullglob  # Prevents issues if no files match
for PARAMS_FILE in 2026_0328*.json; do
    echo "Running training for $PARAMS_FILE"
    HF_DATASETS_DOWNLOADED_DATASETS_PATH="/data/apassi1/datasets/" \
    python "/home/apassi1/deepbottleneck/training.py" "$NOTEBOOK_DIR/$PARAMS_FILE"
done

