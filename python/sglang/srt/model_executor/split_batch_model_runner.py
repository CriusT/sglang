# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Split batch model runner implementation."""

import logging
from enum import Enum
from typing import List, Optional, Tuple, Dict

import torch
import asyncio 

from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class SplitBatchModelRunner(ModelRunner):
    """A model runner that splits input batches into sub-batches for processing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_split_batch = True

    