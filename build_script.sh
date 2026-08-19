#!/bin/sh

set -eu

src_dir="$(cd "$(dirname "$0")" && pwd)"
ftk_dir="${src_dir}/external/ftk"
ftk_commit="78c1e80aa71b52f852eacfdbcea44129e62f3641"
build_jobs="${BUILD_JOBS:-4}"

mkdir -p "${src_dir}/external"

if [ ! -d "${ftk_dir}/.git" ]; then
    git clone --branch eval-cpSZ --single-branch \
        https://github.com/lxAltria/ftk.git "${ftk_dir}"
elif [ ! -f "${ftk_dir}/CMakeLists.txt" ]; then
    echo "Error: ${ftk_dir} exists but is not a valid FTK checkout." >&2
    exit 1
fi

if ! git -C "${ftk_dir}" cat-file -e "${ftk_commit}^{commit}" 2>/dev/null; then
    git -C "${ftk_dir}" fetch origin eval-cpSZ
fi
git -C "${ftk_dir}" checkout --detach "${ftk_commit}"

echo "Building and installing FTK..."
cmake -S "${ftk_dir}" -B "${ftk_dir}/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${ftk_dir}/install"
cmake --build "${ftk_dir}/build" --parallel "${build_jobs}"
cmake --install "${ftk_dir}/build"

# This pinned FTK revision installs FTKTargets.cmake but its package config
# includes ftkTargets.cmake. Provide the lowercase compatibility filename.
ftk_cmake_dir="${ftk_dir}/install/lib/cmake/FTK"
cmake -E copy_if_different \
    "${ftk_cmake_dir}/FTKTargets.cmake" \
    "${ftk_cmake_dir}/ftkTargets.cmake"

echo "Building FaCTz-B..."
cmake -S "${src_dir}" -B "${src_dir}/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DFTK_DIR="${ftk_cmake_dir}"
cmake --build "${src_dir}/build" --parallel "${build_jobs}"

echo "Build completed: ${src_dir}/build/bin"
