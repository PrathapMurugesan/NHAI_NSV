#!/usr/bin/env bash
# Script to run DLM pipeline for a batch of UPCs with S3/EC2 runtime settings.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Load environment defaults if .env exists
if [ -f ".env" ]; then
    # shellcheck disable=SC1091
    source .env
fi

# Override with positional arguments if provided
ZONE="${1:-$ZONE}"
PHASE="${2:-$PHASE}"
CYCLE="${3:-$CYCLE}"
BATCH_FILE="${4:-batches/upcs_4.txt}"
DATA_ROOT="${5:-$DATA_ROOT}"

if [ -z "$ZONE" ] || [ -z "$PHASE" ] || [ -z "$CYCLE" ] || [ -z "$BATCH_FILE" ]; then
    echo "Usage: $0 [ZONE] [PHASE] [CYCLE] [BATCH_FILE] [DATA_ROOT]"
    echo "Example: $0 B 2 1 batches/upcs_4.txt"
    exit 1
fi

echo "========================================================================"
echo "DLM Extraction Pipeline - Batch Execution"
echo "Batch file: $BATCH_FILE"
echo "Zone:       $ZONE | Phase: $PHASE | Cycle: $CYCLE"
echo "Region:     $AWS_REGION | Bucket: $BUCKET"
echo "Output:     $SILVER_S3_ROOT"
echo "========================================================================"

CMD="python3 main.py --zone \"$ZONE\" --phase \"$PHASE\" --cycle \"$CYCLE\" --batch-file \"$BATCH_FILE\""
if [ -n "$DATA_ROOT" ]; then
    CMD="$CMD --data-root \"$DATA_ROOT\""
fi

echo "Running command: $CMD"
eval "$CMD"
