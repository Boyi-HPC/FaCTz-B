#!/bin/sh

set -eu

usage()
{
    cat <<'EOF'
Usage: ./build_script.sh [build|clean|rebuild|help]

Actions:
  build    Configure and build the complete project (default).
  clean    Run CMake's clean target for the configured build directory.
  rebuild  Clean, then configure and build the complete project.
  help     Show this message.

Environment overrides:
  BUILD_DIR             CMake build directory (default: <repo>/build)
  BUILD_TYPE            CMake build type (default: Release)
  BUILD_JOBS            Parallel build jobs (default: number of CPUs)
  CUDACXX                Path to nvcc
  CUDA_ARCHITECTURES     CMake CUDA architecture list (default: 70;75;80;86)
  NVCOMP_ROOT            nvCOMP root containing include/ and lib64/
  SKIP_FTK_BUILD=1       Reuse an existing external/ftk/install package
EOF
}

action="${1:-build}"
if [ "$#" -gt 1 ]; then
    usage >&2
    exit 2
fi

case "${action}" in
    build|clean|rebuild)
        ;;
    help|-h|--help)
        usage
        exit 0
        ;;
    *)
        echo "Error: unknown action '${action}'." >&2
        usage >&2
        exit 2
        ;;
esac

repo_root="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
build_dir="${BUILD_DIR:-${repo_root}/build}"
build_type="${BUILD_TYPE:-Release}"
cuda_architectures="${CUDA_ARCHITECTURES:-70;75;80;86}"

if command -v nproc >/dev/null 2>&1; then
    default_jobs="$(nproc)"
else
    default_jobs=4
fi
build_jobs="${BUILD_JOBS:-${default_jobs}}"

if [ "${action}" = "clean" ] || [ "${action}" = "rebuild" ]; then
    if [ -f "${build_dir}/CMakeCache.txt" ]; then
        echo "Cleaning ${build_dir}..."
        cmake --build "${build_dir}" --target clean
    else
        echo "Nothing to clean: ${build_dir} is not configured."
    fi
    if [ "${action}" = "clean" ]; then
        exit 0
    fi
fi

for tool in cmake git; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "Error: required command '${tool}' was not found." >&2
        exit 1
    fi
done

cuda_compiler="${CUDACXX:-}"
if [ -z "${cuda_compiler}" ]; then
    cuda_compiler="$(command -v nvcc 2>/dev/null || true)"
fi
if [ -z "${cuda_compiler}" ] && [ -x /usr/local/cuda-12/bin/nvcc ]; then
    cuda_compiler=/usr/local/cuda-12/bin/nvcc
fi
if [ -z "${cuda_compiler}" ] || [ ! -x "${cuda_compiler}" ]; then
    echo "Error: nvcc was not found. Set CUDACXX=/path/to/nvcc." >&2
    exit 1
fi

ftk_source_dir="${repo_root}/external/ftk"
ftk_install_dir="${ftk_source_dir}/install"
ftk_cmake_dir="${ftk_install_dir}/lib/cmake/FTK"
ftk_commit="78c1e80aa71b52f852eacfdbcea44129e62f3641"

if [ "${SKIP_FTK_BUILD:-0}" != "1" ]; then
    if [ ! -e "${ftk_source_dir}" ]; then
        mkdir -p "${repo_root}/external"
        git clone --branch eval-cpSZ --single-branch \
            https://github.com/lxAltria/ftk.git "${ftk_source_dir}"
    elif [ ! -d "${ftk_source_dir}/.git" ]; then
        echo "Error: ${ftk_source_dir} exists but is not a Git checkout." >&2
        exit 1
    fi

    if ! git -C "${ftk_source_dir}" cat-file -e "${ftk_commit}^{commit}" 2>/dev/null; then
        git -C "${ftk_source_dir}" fetch origin eval-cpSZ
    fi

    current_ftk_commit="$(git -C "${ftk_source_dir}" rev-parse HEAD)"
    if [ "${current_ftk_commit}" != "${ftk_commit}" ]; then
        if [ -n "$(git -C "${ftk_source_dir}" status --porcelain)" ]; then
            echo "Error: FTK has local changes and is not at the required revision." >&2
            echo "Commit or preserve those changes before rerunning this script." >&2
            exit 1
        fi
        git -C "${ftk_source_dir}" checkout --detach "${ftk_commit}"
    fi

    echo "Building and installing the pinned FTK dependency..."
    cmake -S "${ftk_source_dir}" -B "${ftk_source_dir}/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${ftk_install_dir}"
    cmake --build "${ftk_source_dir}/build" --parallel "${build_jobs}"
    cmake --install "${ftk_source_dir}/build"

elif [ ! -f "${ftk_cmake_dir}/FTKConfig.cmake" ]; then
    echo "Error: SKIP_FTK_BUILD=1, but ${ftk_cmake_dir}/FTKConfig.cmake is missing." >&2
    exit 1
fi

# This FTK revision refers to a lowercase target file from FTKConfig.cmake.
if [ -f "${ftk_cmake_dir}/FTKTargets.cmake" ]; then
    cmake -E copy_if_different \
        "${ftk_cmake_dir}/FTKTargets.cmake" \
        "${ftk_cmake_dir}/ftkTargets.cmake"
fi

nvcomp_root="${NVCOMP_ROOT:-}"
if [ -z "${nvcomp_root}" ] && [ -n "${HOME:-}" ]; then
    nvcomp_root="${HOME}/.local/nvcomp-cu12/nvidia/libnvcomp"
fi
if [ -z "${nvcomp_root}" ] || [ ! -f "${nvcomp_root}/include/nvcomp/ans.hpp" ]; then
    echo "Error: nvCOMP headers were not found." >&2
    echo "Set NVCOMP_ROOT to a directory containing include/nvcomp/ans.hpp and lib64/." >&2
    exit 1
fi
if [ ! -f "${nvcomp_root}/lib64/libnvcomp.so" ] && \
   [ ! -f "${nvcomp_root}/lib64/libnvcomp.so.5" ]; then
    echo "Error: nvCOMP library was not found under ${nvcomp_root}/lib64." >&2
    exit 1
fi

echo "Configuring the complete FaCTz-B project..."
cmake -S "${repo_root}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE="${build_type}" \
    -DCMAKE_CUDA_COMPILER="${cuda_compiler}" \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_architectures}" \
    -DFTK_DIR="${ftk_cmake_dir}" \
    -DCUCPSZ_NVCOMP_ROOT="${nvcomp_root}" \
    -DFACTZ_BUILD_CUDA=ON \
    -DFACTZ_BUILD_CUDA_NAIVE=ON \
    -DFACTZ_BUILD_CUDA_V1=ON \
    -DFACTZ_BUILD_CUDA_V1_1=ON \
    -DFACTZ_BUILD_CUDA_V1_2=ON \
    -DFACTZ_BUILD_CUDA_V1_3=ON \
    -DFACTZ_BUILD_CUDA_V1_4=ON \
    -DFACTZ_BUILD_CUDA_V1_5=ON

echo "Building all project targets..."
cmake --build "${build_dir}" --parallel "${build_jobs}"

expected_programs="
cpszg_2d_no_opt_speed
cpszg_2d_v1
cpszg_2d_v1_1
cpszg_2d_v1_2
cpszg_2d_v1_3
decpszg_2d_v1_3
cpszg_2d_v1_4
cpszg_2d_v1_5
decpszg_2d_v1_5
factz_decpszg_2d
factz_verify_cpszg_2d
"

for program in ${expected_programs}; do
    if [ ! -x "${build_dir}/bin/${program}" ]; then
        echo "Error: expected executable was not generated: ${build_dir}/bin/${program}" >&2
        exit 1
    fi
done

echo
echo "Complete build succeeded. CUDA executables are in: ${build_dir}/bin"
printf '  %s\n' ${expected_programs}
echo "See README.md for per-version run commands."
