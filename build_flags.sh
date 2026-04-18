#!/usr/bin/env bash
set -euo pipefail

export ADDITIONAL_DEFINITIONS="${1:-}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBURING_DIR="${ROOT_DIR}/third_party/liburing"
LIBURING_A="${LIBURING_DIR}/src/liburing.a"

if [[ ! -f "${LIBURING_A}" ]]; then
  echo "[build_flags.sh] building bundled liburing..."
  (cd "${LIBURING_DIR}" && ./configure)
  (cd "${LIBURING_DIR}" && make -C src -j"$(nproc)")
fi

mkdir -p "${ROOT_DIR}/build"
cmake -S "${ROOT_DIR}" -B "${ROOT_DIR}/build"
cmake --build "${ROOT_DIR}/build" -j"$(nproc)"
