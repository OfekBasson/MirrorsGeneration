# TODO: Reorder imports
from diffusers.models.attention import Attention
from typing import Union, List, Dict, Any, Tuple, Callable
import torch
import torch.nn.functional as F
import PIL
from IPython.display import display
import inspect
from diffusers.utils import USE_PEFT_BACKEND
from diffusers.loaders.lora_pipeline import StableDiffusionXLLoraLoaderMixin
from diffusers.loaders.textual_inversion import TextualInversionLoaderMixin
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.models import Transformer2DModel
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import math
from diffusers.models.attention_processor import AttnProcessor2_0
import re

def new_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        r"""
        The forward method of the `Attention` class.

        Args:
            hidden_states (`torch.Tensor`):
                The hidden states of the query.
            encoder_hidden_states (`torch.Tensor`, *optional*):
                The hidden states of the encoder.
            attention_mask (`torch.Tensor`, *optional*):
                The attention mask to use. If `None`, no mask is applied.
            **cross_attention_kwargs:
                Additional keyword arguments to pass along to the cross attention.

        Returns:
            `torch.Tensor`: The output of the attention layer.
        """
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty
        print(f'encoder_hidden_states shape is: {encoder_hidden_states.shape} and its:\n{encoder_hidden_states}')
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        quiet_attn_parameters = {"ip_adapter_masks"}
        unused_kwargs = [
            k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters and k not in quiet_attn_parameters
        ]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"cross_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}
        
        self.processor.__call__ = new_processor_call.__get__(self.processor, AttnProcessor2_0)
        return self.processor.__call__(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

def new_processor_call(
    self,
    attn: Attention,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    temb: Optional[torch.Tensor] = None,
    *args,
    **kwargs,
    ) -> torch.Tensor:
    if len(args) > 0 or kwargs.get("scale", None) is not None:
        deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
        deprecate("scale", "1.0.0", deprecation_message)

    residual = hidden_states
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )

    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        # scaled_dot_product_attention expects attention_mask shape to be
        # (batch, heads, source_length, target_length)
        attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = attn.to_q(hidden_states)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    # the output of sdp = (batch, num_heads, seq_len, head_dim)
    # TODO: add support for attn.scale when we move to Torch 2.1
    hidden_states, calculated_concatenated_attention_maps = custom_scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False, concatenated_attention_maps=self.concatenated_attention_maps
    )

    self.concatenated_attention_maps = calculated_concatenated_attention_maps

    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor

    return hidden_states

def custom_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, concatenated_attention_maps=None):
    # 1. Compute the dot product between query and key (transposed)
    d_k = query.size(-1)  # Get the dimensionality of the key
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    # 2. Apply mask (if provided)
    if attn_mask is not None:
        scores = scores.masked_fill(attn_mask == 0, float('-inf'))

    # 3. Apply softmax to normalize the attention scores
    attention_weights = F.softmax(scores, dim=-1)
    current_attention_map_mean_over_heads = np.expand_dims(attention_weights[1].mean(dim=0).cpu().numpy(), axis=0)
    if concatenated_attention_maps is None:
      concatenated_attention_maps = current_attention_map_mean_over_heads
    else:
      concatenated_attention_maps = np.concatenate((concatenated_attention_maps, current_attention_map_mean_over_heads), axis=0)

    if concatenated_attention_maps.shape[0] == 50:
      average_concatenated_attention_maps_over_all_timesteps = concatenated_attention_maps.mean(axis=0)
      image_resolution_height_and_width = int(math.sqrt(average_concatenated_attention_maps_over_all_timesteps.shape[0]))
      images = [average_concatenated_attention_maps_over_all_timesteps[:, i].reshape(image_resolution_height_and_width, image_resolution_height_and_width) for i in range(77)]
      fig, axes = plt.subplots(11, 7, figsize=(14, 22))
      fig.suptitle("Attention Maps")
      axes = axes.flatten()
      for i in range(77):
        ax = axes[i]
        ax.imshow(images[i], cmap='viridis')

      plt.tight_layout(rect=[0, 0, 1, 0.95])
      plt.show()


    # 4. Multiply by the values
    output = torch.matmul(attention_weights, value)

    return output, concatenated_attention_maps


# def replace_example_docstring(example_docstring):
#     def docstring_decorator(fn):
#         func_doc = fn.__doc__
#         lines = func_doc.split("\n")
#         i = 0
#         while i < len(lines) and re.search(r"^\s*Examples?:\s*$", lines[i]) is None:
#             i += 1
#         if i < len(lines):
#             lines[i] = example_docstring
#             func_doc = "\n".join(lines)
#         else:
#             raise ValueError(
#                 f"The function {fn} should have an empty 'Examples:' in its docstring as placeholder, "
#                 f"current docstring is:\n{func_doc}"
#             )
#         fn.__doc__ = func_doc
#         return fn

#     return docstring_decorator


# EXAMPLE_DOC_STRING = """
#     Examples:
#         ```py
#         >>> import torch
#         >>> from diffusers import StableDiffusionXLPipeline

#         >>> pipe = StableDiffusionXLPipeline.from_pretrained(
#         ...     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
#         ... )
#         >>> pipe = pipe.to("cuda")

#         >>> prompt = "a photo of an astronaut riding a horse on mars"
#         >>> image = pipe(prompt).images[0]
#         ```
# """

# PipelineImageInput = Union[
#     PIL.Image.Image,
#     np.ndarray,
#     torch.Tensor,
#     List[PIL.Image.Image],
#     List[np.ndarray],
#     List[torch.Tensor],
# ]

# @torch.no_grad()
# @replace_example_docstring(EXAMPLE_DOC_STRING)
# def new_pipe_call(
#     self,
#     prompt: Union[str, List[str]] = None,
#     prompt_2: Optional[Union[str, List[str]]] = None,
#     height: Optional[int] = None,
#     width: Optional[int] = None,
#     num_inference_steps: int = 50,
#     timesteps: List[int] = None,
#     sigmas: List[float] = None,
#     denoising_end: Optional[float] = None,
#     guidance_scale: float = 5.0,
#     negative_prompt: Optional[Union[str, List[str]]] = None,
#     negative_prompt_2: Optional[Union[str, List[str]]] = None,
#     num_images_per_prompt: Optional[int] = 1,
#     eta: float = 0.0,
#     generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
#     latents: Optional[torch.Tensor] = None,
#     prompt_embeds: Optional[torch.Tensor] = None,
#     negative_prompt_embeds: Optional[torch.Tensor] = None,
#     pooled_prompt_embeds: Optional[torch.Tensor] = None,
#     negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
#     ip_adapter_image: Optional[PipelineImageInput] = None,
#     ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
#     output_type: Optional[str] = "pil",
#     return_dict: bool = True,
#     cross_attention_kwargs: Optional[Dict[str, Any]] = None,
#     guidance_rescale: float = 0.0,
#     original_size: Optional[Tuple[int, int]] = None,
#     crops_coords_top_left: Tuple[int, int] = (0, 0),
#     target_size: Optional[Tuple[int, int]] = None,
#     negative_original_size: Optional[Tuple[int, int]] = None,
#     negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
#     negative_target_size: Optional[Tuple[int, int]] = None,
#     clip_skip: Optional[int] = None,
#     callback_on_step_end: Optional[
#         Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
#     ] = None,
#     callback_on_step_end_tensor_inputs: List[str] = ["latents"],
#     **kwargs,
# ):
#     r"""
#     Function invoked when calling the pipeline for generation.

#     Args:
#         prompt (`str` or `List[str]`, *optional*):
#             The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
#             instead.
#         prompt_2 (`str` or `List[str]`, *optional*):
#             The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
#             used in both text-encoders
#         height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
#             The height in pixels of the generated image. This is set to 1024 by default for the best results.
#             Anything below 512 pixels won't work well for
#             [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
#             and checkpoints that are not specifically fine-tuned on low resolutions.
#         width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
#             The width in pixels of the generated image. This is set to 1024 by default for the best results.
#             Anything below 512 pixels won't work well for
#             [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
#             and checkpoints that are not specifically fine-tuned on low resolutions.
#         num_inference_steps (`int`, *optional*, defaults to 50):
#             The number of denoising steps. More denoising steps usually lead to a higher quality image at the
#             expense of slower inference.
#         timesteps (`List[int]`, *optional*):
#             Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
#             in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
#             passed will be used. Must be in descending order.
#         sigmas (`List[float]`, *optional*):
#             Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
#             their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
#             will be used.
#         denoising_end (`float`, *optional*):
#             When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
#             completed before it is intentionally prematurely terminated. As a result, the returned sample will
#             still retain a substantial amount of noise as determined by the discrete timesteps selected by the
#             scheduler. The denoising_end parameter should ideally be utilized when this pipeline forms a part of a
#             "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
#             Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output)
#         guidance_scale (`float`, *optional*, defaults to 5.0):
#             Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
#             `guidance_scale` is defined as `w` of equation 2. of [Imagen
#             Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
#             1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
#             usually at the expense of lower image quality.
#         negative_prompt (`str` or `List[str]`, *optional*):
#             The prompt or prompts not to guide the image generation. If not defined, one has to pass
#             `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
#             less than `1`).
#         negative_prompt_2 (`str` or `List[str]`, *optional*):
#             The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
#             `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
#         num_images_per_prompt (`int`, *optional*, defaults to 1):
#             The number of images to generate per prompt.
#         eta (`float`, *optional*, defaults to 0.0):
#             Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
#             [`schedulers.DDIMScheduler`], will be ignored for others.
#         generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
#             One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
#             to make generation deterministic.
#         latents (`torch.Tensor`, *optional*):
#             Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
#             generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
#             tensor will ge generated by sampling using the supplied random `generator`.
#         prompt_embeds (`torch.Tensor`, *optional*):
#             Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
#             provided, text embeddings will be generated from `prompt` input argument.
#         negative_prompt_embeds (`torch.Tensor`, *optional*):
#             Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
#             weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
#             argument.
#         pooled_prompt_embeds (`torch.Tensor`, *optional*):
#             Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
#             If not provided, pooled text embeddings will be generated from `prompt` input argument.
#         negative_pooled_prompt_embeds (`torch.Tensor`, *optional*):
#             Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
#             weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
#             input argument.
#         ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
#         ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
#             Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
#             IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
#             contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
#             provided, embeddings are computed from the `ip_adapter_image` input argument.
#         output_type (`str`, *optional*, defaults to `"pil"`):
#             The output format of the generate image. Choose between
#             [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
#         return_dict (`bool`, *optional*, defaults to `True`):
#             Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
#             of a plain tuple.
#         cross_attention_kwargs (`dict`, *optional*):
#             A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
#             `self.processor` in
#             [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
#         guidance_rescale (`float`, *optional*, defaults to 0.0):
#             Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
#             Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
#             [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
#             Guidance rescale factor should fix overexposure when using zero terminal SNR.
#         original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
#             If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
#             `original_size` defaults to `(height, width)` if not specified. Part of SDXL's micro-conditioning as
#             explained in section 2.2 of
#             [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
#         crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
#             `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
#             `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
#             `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
#             [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
#         target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
#             For most cases, `target_size` should be set to the desired height and width of the generated image. If
#             not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in
#             section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
#         negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
#             To negatively condition the generation process based on a specific image resolution. Part of SDXL's
#             micro-conditioning as explained in section 2.2 of
#             [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
#             information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
#         negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
#             To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
#             micro-conditioning as explained in section 2.2 of
#             [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
#             information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
#         negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
#             To negatively condition the generation process based on a target image resolution. It should be as same
#             as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
#             [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
#             information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
#         callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
#             A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
#             each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
#             DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
#             list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
#         callback_on_step_end_tensor_inputs (`List`, *optional*):
#             The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
#             will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
#             `._callback_tensor_inputs` attribute of your pipeline class.

#     Examples:

#     Returns:
#         [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] or `tuple`:
#         [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
#         `tuple`. When returning a tuple, the first element is a list with the generated images.
#     """
#     callback = kwargs.pop("callback", None)
#     callback_steps = kwargs.pop("callback_steps", None)

#     if callback is not None:
#         deprecate(
#             "callback",
#             "1.0.0",
#             "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
#         )
#     if callback_steps is not None:
#         deprecate(
#             "callback_steps",
#             "1.0.0",
#             "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
#         )

#     if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
#         callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

#     # 0. Default height and width to unet
#     height = height or self.default_sample_size * self.vae_scale_factor
#     width = width or self.default_sample_size * self.vae_scale_factor

#     original_size = original_size or (height, width)
#     target_size = target_size or (height, width)

#     # 1. Check inputs. Raise error if not correct
#     self.check_inputs(
#         prompt,
#         prompt_2,
#         height,
#         width,
#         callback_steps,
#         negative_prompt,
#         negative_prompt_2,
#         prompt_embeds,
#         negative_prompt_embeds,
#         pooled_prompt_embeds,
#         negative_pooled_prompt_embeds,
#         ip_adapter_image,
#         ip_adapter_image_embeds,
#         callback_on_step_end_tensor_inputs,
#     )

#     self._guidance_scale = guidance_scale
#     self._guidance_rescale = guidance_rescale
#     self._clip_skip = clip_skip
#     self._cross_attention_kwargs = cross_attention_kwargs
#     self._denoising_end = denoising_end
#     self._interrupt = False

#     # 2. Define call parameters
#     if prompt is not None and isinstance(prompt, str):
#         batch_size = 1
#     elif prompt is not None and isinstance(prompt, list):
#         batch_size = len(prompt)
#     else:
#         batch_size = prompt_embeds.shape[0]

#     device = self._execution_device

#     # 3. Encode input prompt
#     lora_scale = (
#         self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
#     )

#     (
#         prompt_embeds,
#         negative_prompt_embeds,
#         pooled_prompt_embeds,
#         negative_pooled_prompt_embeds,
#     ) = self.encode_prompt(
#         prompt=prompt,
#         prompt_2=prompt_2,
#         device=device,
#         num_images_per_prompt=num_images_per_prompt,
#         do_classifier_free_guidance=self.do_classifier_free_guidance,
#         negative_prompt=negative_prompt,
#         negative_prompt_2=negative_prompt_2,
#         prompt_embeds=prompt_embeds,
#         negative_prompt_embeds=negative_prompt_embeds,
#         pooled_prompt_embeds=pooled_prompt_embeds,
#         negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
#         lora_scale=lora_scale,
#         clip_skip=self.clip_skip,
#     )

#     # 4. Prepare timesteps
#     timesteps, num_inference_steps = retrieve_timesteps(
#         self.scheduler, num_inference_steps, device, timesteps, sigmas
#     )

#     # 5. Prepare latent variables
#     num_channels_latents = self.unet.config.in_channels
#     latents = self.prepare_latents(
#         batch_size * num_images_per_prompt,
#         num_channels_latents,
#         height,
#         width,
#         prompt_embeds.dtype,
#         device,
#         generator,
#         latents,
#     )

#     # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
#     extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

#     # 7. Prepare added time ids & embeddings
#     add_text_embeds = pooled_prompt_embeds
#     if self.text_encoder_2 is None:
#         text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
#     else:
#         text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

#     add_time_ids = self._get_add_time_ids(
#         original_size,
#         crops_coords_top_left,
#         target_size,
#         dtype=prompt_embeds.dtype,
#         text_encoder_projection_dim=text_encoder_projection_dim,
#     )
#     if negative_original_size is not None and negative_target_size is not None:
#         negative_add_time_ids = self._get_add_time_ids(
#             negative_original_size,
#             negative_crops_coords_top_left,
#             negative_target_size,
#             dtype=prompt_embeds.dtype,
#             text_encoder_projection_dim=text_encoder_projection_dim,
#         )
#     else:
#         negative_add_time_ids = add_time_ids

#     if self.do_classifier_free_guidance:
#         prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
#         add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
#         add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

#     prompt_embeds = prompt_embeds.to(device)
#     add_text_embeds = add_text_embeds.to(device)
#     add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

#     if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
#         image_embeds = self.prepare_ip_adapter_image_embeds(
#             ip_adapter_image,
#             ip_adapter_image_embeds,
#             device,
#             batch_size * num_images_per_prompt,
#             self.do_classifier_free_guidance,
#         )

#     # 8. Denoising loop
#     num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

#     # 8.1 Apply denoising_end
#     if (
#         self.denoising_end is not None
#         and isinstance(self.denoising_end, float)
#         and self.denoising_end > 0
#         and self.denoising_end < 1
#     ):
#         discrete_timestep_cutoff = int(
#             round(
#                 self.scheduler.config.num_train_timesteps
#                 - (self.denoising_end * self.scheduler.config.num_train_timesteps)
#             )
#         )
#         num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
#         timesteps = timesteps[:num_inference_steps]

#     # 9. Optionally get Guidance Scale Embedding
#     timestep_cond = None
#     if self.unet.config.time_cond_proj_dim is not None:
#         guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
#         timestep_cond = self.get_guidance_scale_embedding(
#             guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
#         ).to(device=device, dtype=latents.dtype)

#     self._num_timesteps = len(timesteps)
#     with self.progress_bar(total=num_inference_steps) as progress_bar:
#         for i, t in enumerate(timesteps):
#             if self.interrupt:
#                 continue

#             # expand the latents if we are doing classifier free guidance
#             latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

#             latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

#             # predict the noise residual
#             added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
#             if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
#                 added_cond_kwargs["image_embeds"] = image_embeds
#             noise_pred = self.unet(
#                 latent_model_input,
#                 t,
#                 encoder_hidden_states=prompt_embeds,
#                 timestep_cond=timestep_cond,
#                 cross_attention_kwargs=self.cross_attention_kwargs,
#                 added_cond_kwargs=added_cond_kwargs,
#                 return_dict=False,
#             )[0]

#             # perform guidance
#             if self.do_classifier_free_guidance:
#                 noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#                 noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

#             if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
#                 # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
#                 noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

#             # compute the previous noisy sample x_t -> x_t-1
#             latents_dtype = latents.dtype
#             latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
#             if latents.dtype != latents_dtype:
#                 if torch.backends.mps.is_available():
#                     # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
#                     latents = latents.to(latents_dtype)

#             if callback_on_step_end is not None:
#                 callback_kwargs = {}
#                 for k in callback_on_step_end_tensor_inputs:
#                     callback_kwargs[k] = locals()[k]
#                 callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

#                 latents = callback_outputs.pop("latents", latents)
#                 prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
#                 negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
#                 add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
#                 negative_pooled_prompt_embeds = callback_outputs.pop(
#                     "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
#                 )
#                 add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
#                 negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", negative_add_time_ids)

#             # call the callback, if provided
#             if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
#                 progress_bar.update()
#                 if callback is not None and i % callback_steps == 0:
#                     step_idx = i // getattr(self.scheduler, "order", 1)
#                     callback(step_idx, t, latents)

#             if XLA_AVAILABLE:
#                 xm.mark_step()

#     if not output_type == "latent":
#         # make sure the VAE is in float32 mode, as it overflows in float16
#         needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

#         if needs_upcasting:
#             self.upcast_vae()
#             latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
#         elif latents.dtype != self.vae.dtype:
#             if torch.backends.mps.is_available():
#                 # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
#                 self.vae = self.vae.to(latents.dtype)

#         # unscale/denormalize the latents
#         # denormalize with the mean and std if available and not None
#         has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
#         has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
#         if has_latents_mean and has_latents_std:
#             latents_mean = (
#                 torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
#             )
#             latents_std = (
#                 torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
#             )
#             latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
#         else:
#             latents = latents / self.vae.config.scaling_factor

#         image = self.vae.decode(latents, return_dict=False)[0]

#         # cast back to fp16 if needed
#         if needs_upcasting:
#             self.vae.to(dtype=torch.float16)
#     else:
#         image = latents

#     if not output_type == "latent":
#         # apply watermark if available
#         if self.watermark is not None:
#             image = self.watermark.apply_watermark(image)

#         image = self.image_processor.postprocess(image, output_type=output_type)

#     # Offload all models
#     self.maybe_free_model_hooks()

#     if not return_dict:
#         return (image,)

#     return StableDiffusionXLPipelineOutput(images=image)

def new_encode_prompt(
    self,
    prompt: str,
    prompt_2: Optional[str] = None,
    device: Optional[torch.device] = None,
    num_images_per_prompt: int = 1,
    do_classifier_free_guidance: bool = True,
    negative_prompt: Optional[str] = None,
    negative_prompt_2: Optional[str] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    pooled_prompt_embeds: Optional[torch.Tensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
    lora_scale: Optional[float] = None,
    clip_skip: Optional[int] = None,
):
    r"""
    Encodes the prompt into text encoder hidden states.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            prompt to be encoded
        prompt_2 (`str` or `List[str]`, *optional*):
            The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
            used in both text-encoders
        device: (`torch.device`):
            torch device
        num_images_per_prompt (`int`):
            number of images that should be generated per prompt
        do_classifier_free_guidance (`bool`):
            whether to use classifier free guidance or not
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        negative_prompt_2 (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
            `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
        prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        pooled_prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
            If not provided, pooled text embeddings will be generated from `prompt` input argument.
        negative_pooled_prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
            input argument.
        lora_scale (`float`, *optional*):
            A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        clip_skip (`int`, *optional*):
            Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
            the output of the pre-final layer will be used for computing the prompt embeddings.
    """
    device = device or self._execution_device

    # set lora scale so that monkey patched LoRA
    # function of text encoder can correctly access it
    if lora_scale is not None and isinstance(self, StableDiffusionXLLoraLoaderMixin):
        self._lora_scale = lora_scale

        # dynamically adjust the LoRA scale
        if self.text_encoder is not None:
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder_2, lora_scale)
            else:
                scale_lora_layers(self.text_encoder_2, lora_scale)

    prompt = [prompt] if isinstance(prompt, str) else prompt

    if prompt is not None:
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    # Define tokenizers and text encoders
    tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
    text_encoders = (
        [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
    )

    if prompt_embeds is None:
        prompt_2 = prompt_2 or prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

        # textual inversion: process multi-vector tokens if necessary
        prompt_embeds_list = []
        prompts = [prompt, prompt_2]
        for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, tokenizer)

            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            print(f"tokenized text_inputs is: {text_inputs}")
            text_input_ids = text_inputs.input_ids
            print(f"text_input_ids is: {text_input_ids}")
            untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {tokenizer.model_max_length} tokens: {removed_text}"
                )

            prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
            
            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            if clip_skip is None:
                prompt_embeds = prompt_embeds.hidden_states[-2]
            else:
                # "2" because SDXL always indexes from the penultimate layer.
                prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

    # get unconditional embeddings for classifier free guidance
    zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
    if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
    elif do_classifier_free_guidance and negative_prompt_embeds is None:
        negative_prompt = negative_prompt or ""
        negative_prompt_2 = negative_prompt_2 or negative_prompt

        # normalize str to list
        negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
        negative_prompt_2 = (
            batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
        )

        uncond_tokens: List[str]
        if prompt is not None and type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )
        else:
            uncond_tokens = [negative_prompt, negative_prompt_2]

        negative_prompt_embeds_list = []
        for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
            if isinstance(self, TextualInversionLoaderMixin):
                negative_prompt = self.maybe_convert_prompt(negative_prompt, tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            negative_prompt_embeds = text_encoder(
                uncond_input.input_ids.to(device),
                output_hidden_states=True,
            )
            # We are only ALWAYS interested in the pooled output of the final text encoder
            negative_pooled_prompt_embeds = negative_prompt_embeds[0]
            negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

            negative_prompt_embeds_list.append(negative_prompt_embeds)

        negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

    if self.text_encoder_2 is not None:
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
    else:
        prompt_embeds = prompt_embeds.to(dtype=self.unet.dtype, device=device)

    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    if do_classifier_free_guidance:
        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]

        if self.text_encoder_2 is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        else:
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.unet.dtype, device=device)

        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
        bs_embed * num_images_per_prompt, -1
    )
    if do_classifier_free_guidance:
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )

    if self.text_encoder is not None:
        if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(self.text_encoder, lora_scale)

    if self.text_encoder_2 is not None:
        if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(self.text_encoder_2, lora_scale)
    print(f"Returned prompt shape is: {prompt_embeds.shape} and the values are\n{prompt_embeds}")
    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds
