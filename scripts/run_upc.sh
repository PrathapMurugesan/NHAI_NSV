#!/usr/bin/env bash
# Script to run DLM pipeline for a single UPC with S3/EC2 runtime settings.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# Load environment defaults if .env exists
if [ -f ".env" ]; then
    # shellcheck disable=SC1091
    source .env
fi

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: $0 [ZONE] [PHASE] [CYCLE] [UPC] [DATA_ROOT]"
    echo "Example: $0 B 2 1 N_07024_01001_GJ"
    exit 0
fi

# Override with positional arguments if provided
ZONE="${1:-$ZONE}"
PHASE="${2:-$PHASE}"
CYCLE="${3:-$CYCLE}"
UPC="${4:-$UPC}"
DATA_ROOT="${5:-$DATA_ROOT}"

if [ -z "$ZONE" ] || [ -z "$PHASE" ] || [ -z "$CYCLE" ] || [ -z "$UPC" ]; then
    echo "Usage: $0 [ZONE] [PHASE] [CYCLE] [UPC] [DATA_ROOT]"
    echo "Or configure ZONE, PHASE, CYCLE, UPC in .env"
    echo "Example: $0 B 2 1 N_07024_01001_GJ"
    exit 1
fi

# Resolve per-run partition paths for the current UPC
SILVER_UPC_ROOT="${SILVER_S3_ROOT}/Zone=${ZONE}/Phase=${PHASE}/Cycle=${CYCLE}/UPC=${UPC}"
SILVER_DATA_OUTPUT_DIR="${SILVER_UPC_ROOT}/dlm_kpi_new"
LOG_DIR="${WORK_DIR}/logs"
RUN_LOG_S3="${LOG_DIR}/${UPC}_P${PHASE}_C${CYCLE}.log"

echo "========================================================================"
echo "DLM Extraction Pipeline - UPC Execution"
echo "UPC:    $UPC"
echo "Zone:   $ZONE | Phase: $PHASE | Cycle: $CYCLE"
echo "Region: $AWS_REGION | Bucket: $BUCKET"
echo "Output: $SILVER_UPC_ROOT"
echo "========================================================================"

CMD="python3 main.py --zone \"$ZONE\" --phase \"$PHASE\" --cycle \"$CYCLE\" --upc \"$UPC\""
if [ -n "$DATA_ROOT" ]; then
    CMD="$CMD --data-root \"$DATA_ROOT\""
fi

echo "Running command: $CMD"
eval "$CMD"