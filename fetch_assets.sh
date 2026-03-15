#!/bin/bash
# fetch_assets.sh — Downloads Quake 1 shareware PAK for path tracing demo content
#
# The Quake 1 shareware episode (E1M1–E1M4) is legally redistributable.
# This script downloads it from a well-known mirror.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ASSETS_DIR="$SCRIPT_DIR/assets/maps"
PAK_FILE="$ASSETS_DIR/pak0.pak"

mkdir -p "$ASSETS_DIR"

if [ -f "$PAK_FILE" ]; then
    echo "✓ pak0.pak already exists at: $PAK_FILE"
    echo "  To re-download, delete it first."
    exit 0
fi

echo "Downloading Quake 1 shareware data..."
echo ""

# Quake shareware quake106.zip (approx 9MB) from quaddicted.com
# Contains the legally redistributable shareware version with Episode 1.
# The zip contains a legacy LZH-compressed installer; we extract pak0.pak from it.
QUAKE_URL="https://www.quaddicted.com/files/idgames2/idstuff/quake/quake106.zip"
TEMP_DIR=$(mktemp -d)

cleanup() { rm -rf "$TEMP_DIR"; }
trap cleanup EXIT

if command -v curl &> /dev/null; then
    echo "Downloading quake106.zip with curl..."
    curl -L -o "$TEMP_DIR/quake106.zip" "$QUAKE_URL" --progress-bar
elif command -v wget &> /dev/null; then
    echo "Downloading quake106.zip with wget..."
    wget -O "$TEMP_DIR/quake106.zip" "$QUAKE_URL" --show-progress
else
    echo "ERROR: Neither curl nor wget found. Please install one of them."
    exit 1
fi

echo "Extracting pak0.pak from quake106.zip..."

# Unzip the shareware archive
unzip -o "$TEMP_DIR/quake106.zip" -d "$TEMP_DIR/quake_extract" > /dev/null

# The archive contains resource.1 which is LZH-compressed and holds id1/pak0.pak
RESOURCE_FILE="$TEMP_DIR/quake_extract/resource.1"
if [ ! -f "$RESOURCE_FILE" ]; then
    echo "ERROR: resource.1 not found in quake106.zip"
    exit 1
fi

# Check for lha/lhasa (LZH extraction tool)
if ! command -v lha &> /dev/null; then
    echo "lha not found. Installing lhasa via Homebrew..."
    if command -v brew &> /dev/null; then
        brew install lhasa
    else
        echo "ERROR: lha is required to extract the archive."
        echo "Install it with: brew install lhasa"
        exit 1
    fi
fi

# Extract id1/pak0.pak from the LZH archive
(cd "$TEMP_DIR" && lha x "$RESOURCE_FILE" id1/pak0.pak > /dev/null)

EXTRACTED_PAK="$TEMP_DIR/id1/pak0.pak"
if [ ! -f "$EXTRACTED_PAK" ]; then
    echo "ERROR: Failed to extract pak0.pak from resource.1"
    exit 1
fi

cp "$EXTRACTED_PAK" "$PAK_FILE"

if [ -f "$PAK_FILE" ]; then
    SIZE=$(stat -f%z "$PAK_FILE" 2>/dev/null || stat -c%s "$PAK_FILE" 2>/dev/null || echo "unknown")
    echo ""
    echo "✓ Downloaded pak0.pak ($SIZE bytes) to: $PAK_FILE"
    echo ""
    echo "The shareware PAK contains these maps:"
    echo "  - maps/e1m1.bsp  (The Slipgate Complex)"
    echo "  - maps/e1m2.bsp  (Castle of the Damned)"
    echo "  - maps/e1m3.bsp  (The Necropolis)"
    echo "  - maps/e1m4.bsp  (The Grisly Grotto)"
    echo "  - maps/e1m5.bsp  (Gloom Keep)"
    echo "  - maps/e1m6.bsp  (The Door to Chthon)"
    echo "  - maps/e1m7.bsp  (The House of Chthon)"
    echo "  - maps/e1m8.bsp  (Ziggurat Vertigo)"
    echo ""
    echo "Run the app to load E1M1 automatically."
else
    echo ""
    echo "ERROR: Download failed."
    echo "Please manually download pak0.pak and place it at: $PAK_FILE"
    exit 1
fi
