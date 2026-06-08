#!/usr/bin/env bash
set -euo pipefail

FILEPATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --filepath)
      FILEPATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: lint.sh --filepath path/to/file.py"
      exit 1
      ;;
  esac
done

if [[ -z "$FILEPATH" ]]; then
  echo "Usage: lint.sh --filepath path/to/file.py"
  exit 1
fi

uv run black "$FILEPATH"
uv run isort "$FILEPATH"
uv run ruff check "$FILEPATH" --fix