#!/bin/bash

# OPD continuation stage after OpenThoughts math SFT.
#
# Stage 1:
#   bash examples/on_policy_distillation/run-qwen3-1.7B-sft-openthoughts_math.sh
#
# Stage 2:
#   bash examples/on_policy_distillation/run-qwen3-1.7B-1.7bgrpoteacher-opd_noanswer_dapo_from_sft.sh
#
# This wrapper keeps the original OPD script usable as a baseline while setting
# the SFT checkpoint as the actor load path for the continuation run.

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export OPD_LOAD="${OPD_LOAD:-/root/slime_siqi/output/Qwen3-1.7B_sft_openthoughts_math/}"
export OPD_SAVE="${OPD_SAVE:-/root/slime_siqi/output/Qwen3-1.7B_sft_openthoughts_math_8B_opd_noanswer_dapo/}"

exec bash "${SCRIPT_DIR}/run-qwen3-1.7B-1.7bgrpoteacher-opd_noanswer_dapo.sh" "$@"
