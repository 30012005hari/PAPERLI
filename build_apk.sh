#!/bin/bash
# ============================================
#   Build APK for Research Paper Explainer
#   Run this on Linux or WSL (not Windows)
# ============================================

echo "============================================"
echo "  Building Research Paper Explainer APK"
echo "============================================"

# Install dependencies
pip install buildozer kivy

# Install system deps (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    libffi-dev \
    python3-dev \
    git \
    zip \
    unzip \
    openjdk-17-jdk \
    autoconf \
    libtool \
    pkg-config \
    cmake

cd apk_wrapper

echo ""
echo "Building APK..."
buildozer android debug

echo ""
echo "============================================"
echo "  Build complete!"
echo "  APK: apk_wrapper/bin/*.apk"
echo "============================================"
