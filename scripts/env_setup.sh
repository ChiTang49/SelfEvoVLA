#!/usr/bin/env bash

ROOT="${RIPT_VLA_LITE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [ -d "${ROOT}/env/bin" ]; then
  export PATH="${ROOT}/env/bin:${PATH}"
fi
export PYTHONPATH="${ROOT}/sources/LIBERO:${ROOT}:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH="${ROOT}/config/libero"
export XDG_CACHE_HOME="${ROOT}/tmp/xdg"
export TMPDIR="${ROOT}/tmp"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}"
