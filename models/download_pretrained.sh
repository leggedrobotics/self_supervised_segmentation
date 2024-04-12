#!/bin/bash
#
# Copyright (c) Mark Hamilton. All rights reserved.
# Copyright (c) 2022-2024, ETH Zurich, Piotr Libera, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
# This scripts gets the pretrained weights to run STEGO
#

# From https://stackoverflow.com/a/246128
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

gdown https://drive.google.com/uc?id=1t6dS_9LlN9meN2yoA6iSZEaLlVMEJ4de
