#!/usr/bin/env bash
set -euo pipefail

# Creates two uv environments for benchmarking:
# - baseline: released Keras
# - branch: your feature branch
#
# It installs:
# - identical shared deps in both envs
# - identical JAX stack in both envs
# - different Keras source in each env
#
# Usage:
#   bash setup_envs.sh
#
# Optional overrides:
#   PYTHON_VERSION=3.11 \
#   BASELINE_VENV=.venv_baseline \
#   BRANCH_VENV=.venv_branch \
#   KERAS_BASELINE_SPEC="keras==3.12.0" \
#   KERAS_BRANCH_REF="feat/jax-steps-per-execution-scan" \
#   bash setup_envs.sh

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
BASELINE_VENV="${BASELINE_VENV:-.venv_baseline}"
BRANCH_VENV="${BRANCH_VENV:-.venv_branch}"
ORCH_VENV="${ORCH_VENV:-.venv_orch}"

# Baseline Keras package/version
KERAS_BASELINE_SPEC="${KERAS_BASELINE_SPEC:-keras}"

# Branch install source
KERAS_BRANCH_REPO="${KERAS_BRANCH_REPO:-https://github.com/AmedeoBiolatti/keras.git}"
KERAS_BRANCH_REF="${KERAS_BRANCH_REF:-feat/jax-steps-per-execution-scan}"

# Shared dependencies
COMMON_DEPS=(
  psutil
  numpy
  jax[cuda12_pip]==0.9.1
  tensorflow-datasets==4.9.3
  keras-nlp
  keras-cv
)

# JAX install spec
# Adjust this to match your machine/image.
# Examples:
#   JAX_SPEC="jax"
#   JAX_SPEC='jax[cuda12]'
#   JAX_SPEC='jax[cuda13]'
JAX_SPEC="${JAX_SPEC:-jax}"

# Set to 1 to build wheels locally instead of direct pip install from index/git.
USE_WHEELS="${USE_WHEELS:-0}"

ROOT_DIR="$(pwd)"
BUILD_DIR="${ROOT_DIR}/.bench_build"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

pybin() {
  local venv="$1"
  echo "${venv}/bin/python"
}

install_into() {
  local venv="$1"
  shift
  uv pip install --python "$(pybin "$venv")" "$@"
}

freeze_env() {
  local venv="$1"
  local out="$2"
  "$(pybin "$venv")" -m pip freeze | sort > "$out"
}

print_header() {
  echo
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

build_keras_wheel_from_git() {
  local repo_url="$1"
  local git_ref="$2"
  local workdir="$3"

  rm -rf "$workdir"
  git clone "$repo_url" "$workdir" >/dev/null
  git -C "$workdir" checkout "$git_ref" >/dev/null

  (
    cd "$workdir"
    uv build --wheel >/dev/null
  )

  ls "$workdir"/dist/*.whl | head -n 1
}

main() {
  need_cmd uv
  need_cmd git

  print_header "Creating virtual environments"
  uv venv --python "$PYTHON_VERSION" "$BASELINE_VENV"
  uv venv --python "$PYTHON_VERSION" "$BRANCH_VENV"
  uv venv --python "$PYTHON_VERSION" "$ORCH_VENV"

  print_header "Installing shared dependencies"
  install_into "$BASELINE_VENV" "${COMMON_DEPS[@]}"
  install_into "$BRANCH_VENV" "${COMMON_DEPS[@]}"

  print_header "Installing identical JAX stack"
  install_into "$BASELINE_VENV" "$JAX_SPEC"
  install_into "$BRANCH_VENV" "$JAX_SPEC"

  print_header "Installing orchestrator dependencies"
  install_into "$ORCH_VENV" wandb numpy

  if [[ "$USE_WHEELS" == "1" ]]; then
    print_header "Building Keras wheels"
    mkdir -p "$BUILD_DIR"

    # Baseline wheel
    # If KERAS_BASELINE_SPEC looks like "keras==x.y.z", we install from PyPI directly instead.
    # Wheel build path is mainly intended for git refs, not PyPI specs.
    echo "USE_WHEELS=1 is mainly useful for git-based installs."
    echo "Baseline will still be installed from: $KERAS_BASELINE_SPEC"

    BASELINE_WHEEL_DIR="${BUILD_DIR}/keras_branch_src"
    BRANCH_WHEEL="$(build_keras_wheel_from_git "$KERAS_BRANCH_REPO" "$KERAS_BRANCH_REF" "$BASELINE_WHEEL_DIR")"

    print_header "Installing Keras"
    install_into "$BASELINE_VENV" "$KERAS_BASELINE_SPEC"
    install_into "$BRANCH_VENV" "$BRANCH_WHEEL"
  else
    print_header "Installing Keras"
    install_into "$BASELINE_VENV" "$KERAS_BASELINE_SPEC"
    install_into "$BRANCH_VENV" "git+${KERAS_BRANCH_REPO}@${KERAS_BRANCH_REF}"
  fi

  print_header "Freezing environments"
  freeze_env "$BASELINE_VENV" "${ROOT_DIR}/baseline_freeze.txt"
  freeze_env "$BRANCH_VENV" "${ROOT_DIR}/branch_freeze.txt"

  print_header "Environment summary"
  echo "Baseline python: $(realpath "$(pybin "$BASELINE_VENV")")"
  echo "Branch   python: $(realpath "$(pybin "$BRANCH_VENV")")"
  echo
  echo "Baseline keras:"
  "$(pybin "$BASELINE_VENV")" - <<'PY'
import keras
print(getattr(keras, "__version__", "unknown"))
PY
  echo
  echo "Branch keras:"
  "$(pybin "$BRANCH_VENV")" - <<'PY'
import keras
print(getattr(keras, "__version__", "unknown"))
PY
  echo
  echo "Wrote:"
  echo "  baseline_freeze.txt"
  echo "  branch_freeze.txt"
}

main "$@"