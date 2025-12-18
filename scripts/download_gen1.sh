#!/bin/bash

# Download Gen1 Automotive Detection Dataset
# 
# The Gen1 dataset must be downloaded from Prophesee's website:
# https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/
#
# This script provides instructions and directory setup.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Gen1 Dataset Download Helper${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Default data directory
DATA_DIR="${1:-./data/gen1}"

echo -e "${YELLOW}Target directory: ${DATA_DIR}${NC}"
echo ""

# Create directory structure
echo "Creating directory structure..."
mkdir -p "${DATA_DIR}/train"
mkdir -p "${DATA_DIR}/val"
mkdir -p "${DATA_DIR}/test"
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

echo -e "${YELLOW}IMPORTANT: Manual Download Required${NC}"
echo ""
echo "The Gen1 dataset must be downloaded manually from Prophesee's website."
echo ""
echo "Steps:"
echo "1. Go to: https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/"
echo ""
echo "2. Register/Login to download the dataset"
echo ""
echo "3. Download the following files:"
echo "   - gen1_train.tar (training data)"
echo "   - gen1_test.tar (test data)"
echo ""
echo "4. Extract the archives to ${DATA_DIR}:"
echo "   tar -xf gen1_train.tar -C ${DATA_DIR}/train"
echo "   tar -xf gen1_test.tar -C ${DATA_DIR}/test"
echo ""
echo "Expected directory structure:"
echo "  ${DATA_DIR}/"
echo "  ├── train/"
echo "  │   ├── <sequence_name>/"
echo "  │   │   ├── events.dat or events.h5"
echo "  │   │   └── labels_v2.npy or <sequence>_bbox.npy"
echo "  │   └── ..."
echo "  └── test/"
echo "      ├── <sequence_name>/"
echo "      │   ├── events.dat or events.h5"
echo "      │   └── labels_v2.npy or <sequence>_bbox.npy"
echo "      └── ..."
echo ""

# Check if Metavision SDK is available
echo "Checking for Metavision SDK..."
if python -c "import metavision_core" 2>/dev/null; then
    echo -e "${GREEN}✓ Metavision SDK is installed${NC}"
else
    echo -e "${YELLOW}! Metavision SDK not found${NC}"
    echo ""
    echo "Metavision SDK provides faster event file loading."
    echo "Without it, a fallback parser will be used (slower)."
    echo ""
    echo "To install Metavision SDK (optional):"
    echo "  pip install metavision-sdk-base"
    echo ""
    echo "Or visit: https://docs.prophesee.ai/stable/installation/index.html"
fi
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Setup complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "After downloading and extracting the data, verify with:"
echo "  python -c \"from src.data import Gen1Dataset; d = Gen1Dataset('${DATA_DIR}'); print(f'Loaded {len(d)} samples')\""
