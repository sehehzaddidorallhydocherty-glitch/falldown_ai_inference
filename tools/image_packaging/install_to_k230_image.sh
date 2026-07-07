#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

K230_SDK_ROOT="${K230_SDK_ROOT:-/home/ubuntu/k230_sdk}"
OUTPUT_CONFIG="${OUTPUT_CONFIG:-}"
BUILD_APP="${BUILD_APP:-1}"
BUILD_NOTIFY="${BUILD_NOTIFY:-1}"
ENABLE_LINUX_AUTOSTART="${ENABLE_LINUX_AUTOSTART:-1}"
ENABLE_RTSMART_AUTOSTART="${ENABLE_RTSMART_AUTOSTART:-1}"
MODEL_PATH="${MODEL_PATH:-}"
INSTALL_DIR_NAME="${INSTALL_DIR_NAME:-falldown}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --sdk-root <path>       K230 SDK root. Default: /home/ubuntu/k230_sdk
  --output-config <name>  SDK output config directory name.
                          Default: auto-detect output/<single config>
  --model <path>          best_fp16.kmodel path. Default: search project and /sharefs
  --no-build-app          Skip RT-Smart app build.
  --no-build-notify       Skip Linux notify client build.
  --no-linux-autostart    Do not install Linux init.d autostart script.
  --no-rtsmart-autostart  Do not install RT-Smart autostart helper script.

Environment overrides:
  K230_SDK_ROOT=/home/ubuntu/k230_sdk
  OUTPUT_CONFIG=k230_canmv_dongshanpi_defconfig
  MODEL_PATH=/path/to/best_fp16.kmodel

This script stages files into the K230 SDK output tree before the final SDK
image packaging step. Run it from the project directory copied into the SDK,
then rebuild/package the SDK image from K230_SDK_ROOT.
EOF
}

die() {
    echo "[image-packaging] error: $*" >&2
    exit 1
}

log() {
    echo "[image-packaging] $*"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sdk-root)
            K230_SDK_ROOT="$2"
            shift 2
            ;;
        --output-config)
            OUTPUT_CONFIG="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --no-build-app)
            BUILD_APP=0
            shift
            ;;
        --no-build-notify)
            BUILD_NOTIFY=0
            shift
            ;;
        --no-linux-autostart)
            ENABLE_LINUX_AUTOSTART=0
            shift
            ;;
        --no-rtsmart-autostart)
            ENABLE_RTSMART_AUTOSTART=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

[[ -d "${K230_SDK_ROOT}" ]] || die "K230 SDK root not found: ${K230_SDK_ROOT}"
[[ -f "${PROJECT_DIR}/CMakeLists.txt" ]] || die "project CMakeLists.txt not found: ${PROJECT_DIR}"

detect_output_config() {
    if [[ -n "${OUTPUT_CONFIG}" ]]; then
        echo "${OUTPUT_CONFIG}"
        return
    fi

    local output_dir="${K230_SDK_ROOT}/output"
    [[ -d "${output_dir}" ]] || die "SDK output directory not found: ${output_dir}"

    mapfile -t configs < <(find "${output_dir}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
    if [[ "${#configs[@]}" -eq 1 ]]; then
        echo "${configs[0]}"
        return
    fi

    for candidate in k230_canmv_dongshanpi_defconfig k230_evb_defconfig; do
        if [[ -d "${output_dir}/${candidate}" ]]; then
            echo "${candidate}"
            return
        fi
    done

    die "cannot auto-detect output config; pass --output-config <name>"
}

find_first_dir() {
    local base="$1"
    shift
    local rel
    for rel in "$@"; do
        if [[ -d "${base}/${rel}" ]]; then
            echo "${base}/${rel}"
            return 0
        fi
    done
    return 1
}

find_model() {
    if [[ -n "${MODEL_PATH}" ]]; then
        [[ -f "${MODEL_PATH}" ]] || die "model not found: ${MODEL_PATH}"
        echo "${MODEL_PATH}"
        return
    fi

    local candidate
    for candidate in \
        "${PROJECT_DIR}/best_fp16.kmodel" \
        "${PROJECT_DIR}/models/best_fp16.kmodel" \
        "${PROJECT_DIR}/../best_fp16.kmodel" \
        "/sharefs/best_fp16.kmodel"; do
        if [[ -f "${candidate}" ]]; then
            echo "${candidate}"
            return
        fi
    done

    die "best_fp16.kmodel not found; pass --model /path/to/best_fp16.kmodel"
}

OUTPUT_CONFIG="$(detect_output_config)"
OUTPUT_DIR="${K230_SDK_ROOT}/output/${OUTPUT_CONFIG}"
IMAGES_DIR="${OUTPUT_DIR}/images"
LITTLE_ROOTFS="$(find_first_dir "${OUTPUT_DIR}" \
    "images/little-core/rootfs" \
    "little/buildroot-ext/target" \
    "little/rootfs" || true)"
BIG_ROOTFS="$(find_first_dir "${OUTPUT_DIR}" \
    "images/big-core/rootfs" \
    "big/rootfs" \
    "images/big/rootfs" || true)"

[[ -d "${OUTPUT_DIR}" ]] || die "SDK output config directory not found: ${OUTPUT_DIR}"
[[ -n "${LITTLE_ROOTFS}" ]] || die "little-core rootfs staging directory not found under ${OUTPUT_DIR}"

SHAREFS_DIR="${LITTLE_ROOTFS}/sharefs"
INSTALL_DIR="${SHAREFS_DIR}/${INSTALL_DIR_NAME}"
MODEL_FILE="$(find_model)"

log "project: ${PROJECT_DIR}"
log "sdk root: ${K230_SDK_ROOT}"
log "output config: ${OUTPUT_CONFIG}"
log "little rootfs: ${LITTLE_ROOTFS}"
log "big rootfs: ${BIG_ROOTFS:-not found}"
log "sharefs staging: ${SHAREFS_DIR}"
log "model: ${MODEL_FILE}"

if [[ "${BUILD_APP}" == "1" ]]; then
    log "building RT-Smart fall detection ELFs"
    cmake -S "${PROJECT_DIR}" -B "${PROJECT_DIR}/build"
    cmake --build "${PROJECT_DIR}/build" --parallel "$(nproc)"
fi

if [[ "${BUILD_NOTIFY}" == "1" ]]; then
    log "building Linux little-core notify client"
    make -C "${PROJECT_DIR}/linux_notify" clean
    make -C "${PROJECT_DIR}/linux_notify" K230_SDK_ROOT="${K230_SDK_ROOT}"
fi

required_files=(
    "${PROJECT_DIR}/build/falldown_ai_event.elf"
    "${PROJECT_DIR}/build/falldown_video_display.elf"
    "${PROJECT_DIR}/build/falldown_split_launcher.elf"
    "${PROJECT_DIR}/linux_notify/falldown_notify_client"
)

for file in "${required_files[@]}"; do
    [[ -f "${file}" ]] || die "required build output missing: ${file}"
done

log "installing runtime files to ${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}" "${SHAREFS_DIR}/sdcard/evidence"
install -m 0755 "${PROJECT_DIR}/build/falldown_ai_event.elf" "${INSTALL_DIR}/"
install -m 0755 "${PROJECT_DIR}/build/falldown_video_display.elf" "${INSTALL_DIR}/"
install -m 0755 "${PROJECT_DIR}/build/falldown_split_launcher.elf" "${INSTALL_DIR}/"
install -m 0755 "${PROJECT_DIR}/linux_notify/falldown_notify_client" "${INSTALL_DIR}/"
install -m 0644 "${MODEL_FILE}" "${INSTALL_DIR}/best_fp16.kmodel"

install -m 0755 "${PROJECT_DIR}/build/falldown_ai_event.elf" "${SHAREFS_DIR}/falldown_ai_event.elf"
install -m 0755 "${PROJECT_DIR}/build/falldown_video_display.elf" "${SHAREFS_DIR}/falldown_video_display.elf"
install -m 0755 "${PROJECT_DIR}/build/falldown_split_launcher.elf" "${SHAREFS_DIR}/falldown_split_launcher.elf"
install -m 0755 "${PROJECT_DIR}/linux_notify/falldown_notify_client" "${SHAREFS_DIR}/falldown_notify_client"
install -m 0644 "${MODEL_FILE}" "${SHAREFS_DIR}/best_fp16.kmodel"

cat > "${INSTALL_DIR}/start_rtsmart_falldown.sh" <<'EOF'
#!/bin/sh
cd /sharefs/falldown || exit 1
exec ./falldown_split_launcher.elf best_fp16.kmodel 0
EOF
chmod 0755 "${INSTALL_DIR}/start_rtsmart_falldown.sh"

cat > "${INSTALL_DIR}/start_linux_notify.sh" <<'EOF'
#!/bin/sh
mkdir -p /sharefs/sdcard
cd /sharefs/falldown || exit 1
exec ./falldown_notify_client /sharefs/sdcard/fall_events.log
EOF
chmod 0755 "${INSTALL_DIR}/start_linux_notify.sh"

if [[ "${ENABLE_LINUX_AUTOSTART}" == "1" ]]; then
    INIT_DIR="${LITTLE_ROOTFS}/etc/init.d"
    mkdir -p "${INIT_DIR}"
    cat > "${INIT_DIR}/S99falldown_notify" <<'EOF'
#!/bin/sh

case "$1" in
    start|"")
        echo "[falldown] start Linux notify client"
        mkdir -p /sharefs/sdcard
        (
            sleep 8
            cd /sharefs/falldown || exit 1
            ./falldown_notify_client /sharefs/sdcard/fall_events.log >> /sharefs/sdcard/falldown_notify.log 2>&1
        ) &
        ;;
    stop)
        killall falldown_notify_client 2>/dev/null || true
        ;;
    restart)
        "$0" stop
        "$0" start
        ;;
esac

exit 0
EOF
    chmod 0755 "${INIT_DIR}/S99falldown_notify"
    log "installed Linux autostart: ${INIT_DIR}/S99falldown_notify"
fi

if [[ "${ENABLE_RTSMART_AUTOSTART}" == "1" ]]; then
    cat > "${INSTALL_DIR}/falldown_rtsmart_autostart.msh" <<'EOF'
cd /sharefs/falldown
./falldown_split_launcher.elf best_fp16.kmodel 0
EOF
    log "installed RT-Smart autostart helper: ${INSTALL_DIR}/falldown_rtsmart_autostart.msh"
    log "If your SDK has an RT-Smart auto-command script, call this helper from it."
fi

log "installed files:"
find "${INSTALL_DIR}" -maxdepth 1 -type f -printf '  %f\n' | sort

if [[ -d "${IMAGES_DIR}" ]]; then
    log "next: rebuild/package SDK image from ${K230_SDK_ROOT}; images are expected under ${IMAGES_DIR}"
else
    log "next: rebuild/package SDK image from ${K230_SDK_ROOT}"
fi
