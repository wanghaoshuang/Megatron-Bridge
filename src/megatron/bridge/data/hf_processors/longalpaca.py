# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
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

"""Processing functions for LongAlpaca-12k dataset.

Dataset: https://huggingface.co/datasets/Yukang/LongAlpaca-12k

LongAlpaca-12k is a long-context instruction-following dataset with ~12k examples.
Each example has an ``instruction``, optional ``input``, and ``output`` field
following the standard Alpaca format.
"""

from typing import Any, Optional

from megatron.bridge.data.builders.hf_dataset import ProcessExampleOutput
from megatron.bridge.training.tokenizers.tokenizer import MegatronTokenizer


def process_longalpaca_example(
    example: dict[str, Any], _tokenizer: Optional[MegatronTokenizer] = None
) -> ProcessExampleOutput:
    """Process a single LongAlpaca-12k example into the required format.

    Args:
        example: Raw example containing 'instruction', 'input', and 'output'
        _tokenizer: Optional tokenizer (not used in this processor)

    Returns:
        ProcessExampleOutput with formatted input/output
    """
    instruction = example["instruction"]
    context = example.get("input", "")

    if context:
        _input = f"{instruction}\n\n{context}"
    else:
        _input = instruction

    _output = example["output"]

    return ProcessExampleOutput(input=_input, output=_output, original_answers=[_output])
