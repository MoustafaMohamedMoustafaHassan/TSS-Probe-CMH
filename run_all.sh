#!/bin/bash
# =============================================================================
# TSS: Transferable Stress Signals — End-to-End Pipeline
# =============================================================================
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh [--skip-prep] [--skip-tests] [--channels ABC]
#
# Stages:
#   1. Environment check
#   2. Unit tests (pytest)
#   3. Data preparation (01_prepare_data.py)
#   4. Training + evaluation (02_train_evaluate.py)
#   5. Masking suite (03_masking_suite.py)
#   6. Statistical analysis (04_statistical_analysis.py)
#   7. Advanced analysis (05_advanced_analysis.py)
# =============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Defaults
SKIP_PREP=false
SKIP_TESTS=false
CHANNELS="ABC"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-prep)   SKIP_PREP=true; shift ;;
        --skip-tests)  SKIP_TESTS=true; shift ;;
        --channels)    CHANNELS="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: ./run_all.sh [--skip-prep] [--skip-tests] [--channels ABC]"
            exit 0 ;;
        *) log_error "Unknown argument: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================================="
echo "  TSS: Transferable Stress Signals — Full Pipeline"
echo "  Channels: $CHANNELS"
echo "  Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================================="

# ── Stage 1: Environment ──
log_info "Stage 1: Checking environment..."

python3 -c "
import sys
print(f'  Python: {sys.version}')
for pkg in ['sklearn', 'pandas', 'numpy', 'scipy']:
    try:
        m = __import__(pkg)
        print(f'  {pkg}: {m.__version__}')
    except ImportError:
        print(f'  {pkg}: MISSING — run: pip install -r requirements.txt')
        sys.exit(1)
try:
    import spacy
    nlp = spacy.load('en_core_web_sm')
    print(f'  spaCy: {spacy.__version__} (en_core_web_sm OK)')
except:
    print(f'  spaCy: NOT AVAILABLE (SVO triples will be disabled)')
"

# ── Stage 2: Tests ──
if [ "$SKIP_TESTS" = false ]; then
    log_info "Stage 2: Running unit tests..."
    if command -v pytest &>/dev/null; then
        python3 -m pytest tests/ -v --tb=short 2>&1 | tail -20
    else
        log_warn "pytest not installed — skipping tests"
    fi
else
    log_warn "Stage 2: Tests skipped (--skip-tests)"
fi

# ── Stage 3: Data Preparation ──
if [ "$SKIP_PREP" = false ]; then
    log_info "Stage 3: Data preparation..."
    if [ -f "scripts/01_prepare_data.py" ] && [ -d "data/raw" ]; then
        python3 scripts/01_prepare_data.py \
            --raw_dir data/raw \
            --output_dir data/processed \
            --remove_mh_keywords \
            2>&1 | tail -15
    else
        log_warn "Skipping data prep — no raw data or script"
    fi
else
    log_warn "Stage 3: Data prep skipped (--skip-prep)"
fi

# ── Stage 4: Training + Evaluation ──
log_info "Stage 4: Training + Evaluation..."
if [ -f "scripts/02_train_evaluate.py" ]; then
    python3 scripts/02_train_evaluate.py \
        --data_dir data/processed \
        --output_dir outputs \
        --channels "$CHANNELS" \
        2>&1 | tail -30
fi

# ── Stage 5: Masking Suite ──
log_info "Stage 5: Masking experiments..."
if [ -f "scripts/03_masking_suite.py" ]; then
    python3 scripts/03_masking_suite.py \
        --data_dir data/processed \
        --output_dir outputs/masking \
        --channels "$CHANNELS" \
        2>&1 | tail -20
fi

# ── Stage 6: Statistical Analysis ──
log_info "Stage 6: Statistical analysis..."
RESULTS_FILE=$(ls -t outputs/tss_results_*.json 2>/dev/null | head -1)
if [ -n "$RESULTS_FILE" ] && [ -f "scripts/04_statistical_analysis.py" ]; then
    python3 scripts/04_statistical_analysis.py \
        --results "$RESULTS_FILE" \
        --output_dir outputs/statistics \
        2>&1 | tail -20
fi

# ── Stage 7: Advanced Analysis ──
log_info "Stage 7: Advanced analysis (SHAP, clusters, divergence)..."
if [ -n "$RESULTS_FILE" ] && [ -f "scripts/05_advanced_analysis.py" ]; then
    python3 scripts/05_advanced_analysis.py \
        --results "$RESULTS_FILE" \
        --data_dir data/processed \
        --output_dir outputs/advanced \
        2>&1 | tail -20
fi

echo ""
echo "=================================================================="
log_info "Pipeline complete! Outputs in: outputs/"
echo "=================================================================="
