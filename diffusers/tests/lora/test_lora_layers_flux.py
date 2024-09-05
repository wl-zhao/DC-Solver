# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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
import os
import sys
import tempfile
import unittest

import numpy as np
import safetensors.torch
import torch
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer, T5EncoderModel

from diffusers import FlowMatchEulerDiscreteScheduler, FluxPipeline, FluxTransformer2DModel
from diffusers.utils.testing_utils import floats_tensor, is_peft_available, require_peft_backend, torch_device


if is_peft_available():
    from peft.utils import get_peft_model_state_dict

sys.path.append(".")

from utils import PeftLoraLoaderMixinTests, check_if_lora_correctly_set  # noqa: E402


@require_peft_backend
class FluxLoRATests(unittest.TestCase, PeftLoraLoaderMixinTests):
    pipeline_class = FluxPipeline
    scheduler_cls = FlowMatchEulerDiscreteScheduler()
    scheduler_kwargs = {}
    uses_flow_matching = True
    transformer_kwargs = {
        "patch_size": 1,
        "in_channels": 4,
        "num_layers": 1,
        "num_single_layers": 1,
        "attention_head_dim": 16,
        "num_attention_heads": 2,
        "joint_attention_dim": 32,
        "pooled_projection_dim": 32,
        "axes_dims_rope": [4, 4, 8],
    }
    transformer_cls = FluxTransformer2DModel
    vae_kwargs = {
        "sample_size": 32,
        "in_channels": 3,
        "out_channels": 3,
        "block_out_channels": (4,),
        "layers_per_block": 1,
        "latent_channels": 1,
        "norm_num_groups": 1,
        "use_quant_conv": False,
        "use_post_quant_conv": False,
        "shift_factor": 0.0609,
        "scaling_factor": 1.5035,
    }
    has_two_text_encoders = True
    tokenizer_cls, tokenizer_id = CLIPTokenizer, "peft-internal-testing/tiny-clip-text-2"
    tokenizer_2_cls, tokenizer_2_id = AutoTokenizer, "hf-internal-testing/tiny-random-t5"
    text_encoder_cls, text_encoder_id = CLIPTextModel, "peft-internal-testing/tiny-clip-text-2"
    text_encoder_2_cls, text_encoder_2_id = T5EncoderModel, "hf-internal-testing/tiny-random-t5"

    @property
    def output_shape(self):
        return (1, 8, 8, 3)

    def get_dummy_inputs(self, with_generator=True):
        batch_size = 1
        sequence_length = 10
        num_channels = 4
        sizes = (32, 32)

        generator = torch.manual_seed(0)
        noise = floats_tensor((batch_size, num_channels) + sizes)
        input_ids = torch.randint(1, sequence_length, size=(batch_size, sequence_length), generator=generator)

        pipeline_inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "num_inference_steps": 4,
            "guidance_scale": 0.0,
            "height": 8,
            "width": 8,
            "output_type": "np",
        }
        if with_generator:
            pipeline_inputs.update({"generator": generator})

        return noise, input_ids, pipeline_inputs

    def test_with_alpha_in_state_dict(self):
        components, _, denoiser_lora_config = self.get_dummy_components(FlowMatchEulerDiscreteScheduler)
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        output_no_lora = pipe(**inputs, generator=torch.manual_seed(0)).images
        self.assertTrue(output_no_lora.shape == self.output_shape)

        pipe.transformer.add_adapter(denoiser_lora_config)
        self.assertTrue(check_if_lora_correctly_set(pipe.transformer), "Lora not correctly set in transformer")

        images_lora = pipe(**inputs, generator=torch.manual_seed(0)).images

        with tempfile.TemporaryDirectory() as tmpdirname:
            denoiser_state_dict = get_peft_model_state_dict(pipe.transformer)
            self.pipeline_class.save_lora_weights(tmpdirname, transformer_lora_layers=denoiser_state_dict)

            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors")))
            pipe.unload_lora_weights()
            pipe.load_lora_weights(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors"))

            # modify the state dict to have alpha values following
            # https://huggingface.co/TheLastBen/Jon_Snow_Flux_LoRA/blob/main/jon_snow.safetensors
            state_dict_with_alpha = safetensors.torch.load_file(
                os.path.join(tmpdirname, "pytorch_lora_weights.safetensors")
            )
            alpha_dict = {}
            for k, v in state_dict_with_alpha.items():
                # only do for `transformer` and for the k projections -- should be enough to test.
                if "transformer" in k and "to_k" in k and "lora_A" in k:
                    alpha_dict[f"{k}.alpha"] = float(torch.randint(10, 100, size=()))
            state_dict_with_alpha.update(alpha_dict)

        images_lora_from_pretrained = pipe(**inputs, generator=torch.manual_seed(0)).images
        self.assertTrue(check_if_lora_correctly_set(pipe.transformer), "Lora not correctly set in denoiser")

        pipe.unload_lora_weights()
        pipe.load_lora_weights(state_dict_with_alpha)
        images_lora_with_alpha = pipe(**inputs, generator=torch.manual_seed(0)).images

        self.assertTrue(
            np.allclose(images_lora, images_lora_from_pretrained, atol=1e-3, rtol=1e-3),
            "Loading from saved checkpoints should give same results.",
        )
        self.assertFalse(np.allclose(images_lora_with_alpha, images_lora, atol=1e-3, rtol=1e-3))
