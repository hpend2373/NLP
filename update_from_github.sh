#!/bin/bash

# GitHub에서 최신 정보 가져오기
echo "Fetching latest from GitHub..."
git fetch origin

# 수정된 핵심 파일들만 업데이트
FILES=(
    "ikea_mepnet_adapter/datasets/ikea_dataset.py"
    "ikea_mepnet_adapter/datasets/ikea_dataset_fixed.py"
    "ikea_mepnet_adapter/models/constraints/constraint_engine_final.py"
    "ikea_mepnet_adapter/models/constraints/constraint_engine_complete.py"
    "ikea_mepnet_adapter/scripts/train/train_ikea.py"
    "ikea_mepnet_adapter/datasets/transforms_ikea.py"
    "ikea_mepnet_adapter/datasets/transforms_ikea_fixed.py"
)

echo "Updating files from GitHub..."
for file in "${FILES[@]}"; do
    if git ls-tree origin/main --name-only | grep -q "^$file$"; then
        echo "Updating: $file"
        git checkout origin/main -- "$file"
    else
        echo "Skipping (not found): $file"
    fi
done

echo "Update complete!"
echo ""
echo "Updated files:"
git status --short | grep "^M"