#!/usr/bin/env bash

# Resolve script root path
export UNITREE_RL_LAB_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Choose python from environment (allow override via $PYTHON)
if [[ -z "${PYTHON}" ]]; then
    python_exe=$(command -v python3 || command -v python)
else
    python_exe="${PYTHON}"
fi

if [[ -z "${python_exe}" ]]; then
    echo "[Error] No Python interpreter found. Install python or set the PYTHON variable."
    exit 1
fi

# Autocomplete wrapper for training command
_ut_rl_lab_python_argcomplete_wrapper() {
    local IFS=$'\013'
    local SUPPRESS_SPACE=0
    if compopt +o nospace 2> /dev/null; then
        SUPPRESS_SPACE=1
    fi

    COMPREPLY=( $(IFS="$IFS" \
                    COMP_LINE="$COMP_LINE" \
                    COMP_POINT="$COMP_POINT" \
                    COMP_TYPE="$COMP_TYPE" \
                    _ARGCOMPLETE=1 \
                    _ARGCOMPLETE_SUPPRESS_SPACE=$SUPPRESS_SPACE \
                    ${python_exe} ${UNITREE_RL_LAB_PATH}/scripts/rsl_rl/train.py \
                    8>&1 9>&2 1>/dev/null 2>/dev/null) )
}
complete -o nospace -F _ut_rl_lab_python_argcomplete_wrapper "./unitree_rl_lab.sh"

# Optional: setup environment variables on install (non-conda environments)
_ut_setup_python_env() {
    local env_file="${UNITREE_RL_LAB_PATH}/env_setup.sh"

    printf '%s\n' "#!/usr/bin/env bash" \
        "# Auto-sourced environment for unitree_rl_lab" \
        "export UNITREE_RL_LAB_PATH=${UNITREE_RL_LAB_PATH}" \
        "alias unitree='${UNITREE_RL_LAB_PATH}/unitree_rl_lab.sh'" \
        "" > "$env_file"

    echo "[Info] Created environment setup script:"
    echo "  source ${env_file}"
    echo "Add this line to ~/.bashrc or ~/.zshrc for automatic setup:"
    echo "  source ${env_file}"
}

# CLI dispatch
case "$1" in
    -i|--install)
        git lfs install
        pip install -e ${UNITREE_RL_LAB_PATH}/source/unitree_rl_lab/
        _ut_setup_python_env
        command -v activate-global-python-argcomplete &> /dev/null && activate-global-python-argcomplete
        ;;
    -l|--list)
        shift
        ${python_exe} ${UNITREE_RL_LAB_PATH}/scripts/list_envs.py "$@"
        ;;
    -p|--play)
        shift
        ${python_exe} ${UNITREE_RL_LAB_PATH}/scripts/rsl_rl/play.py "$@"
        ;;
    -t|--train)
        shift
        ${python_exe} ${UNITREE_RL_LAB_PATH}/scripts/rsl_rl/train.py --headless "$@"
        ;;
    *)
        echo "Usage:"
        echo "  $0 --install              Install package and environment helpers"
        echo "  $0 --list <args>          List available training environments"
        echo "  $0 --play <args>          Run in play mode"
        echo "  $0 --train <args>         Train headless"
        ;;
esac