from diffusers.models.attention import Attention
from typing import Optional
from mirrors_helper_functions import custom_scaled_dot_product_attention
import torch

def new_processor_call(
    self,
    attn: Attention,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    temb: Optional[torch.Tensor] = None,
    # TODO: change list to Optional[List] or something?
    # TODO: Isn't it better to save the tokenized_prompt directly inside the processor instead of inside the Attention instance and forward it to the processor?
    tokenized_prompt: list = None,
    module_name: str = "",
    *args,
    **kwargs,
    ) -> torch.Tensor:
    # print(f'Inside "new_processor_call", initial self.pipe.concatenated_attention_maps_over_all_steps_and_attention_modules shape is: {"None" if (not self.pipe or self.pipe.concatenated_attention_maps_over_all_steps_and_attention_modules is None) else self.pipe.concatenated_attention_maps_over_all_steps_and_attention_modules.shape}')
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

    # TODO: add support for attn.scale when we move to Torch 2.1
    hidden_states, calculated_concatenated_attention_maps, concatenated_attention_maps_over_all_steps_and_attention_modules, normalized_attention_map = custom_scaled_dot_product_attention(
        query, 
        key, 
        value, 
        attn_mask=attention_mask, 
        dropout_p=0.0, 
        is_causal=False, 
        concatenated_current_module_attention_maps=self.concatenated_attention_maps, 
        tokenized_prompt=tokenized_prompt,
        module_name = module_name,
        concatenated_attention_maps_over_all_steps_and_attention_modules = self.pipe.concatenated_attention_maps_over_all_steps_and_attention_modules,
        display_option = self.display_option,
        module_to_display = self.module_to_display
    )
    self.concatenated_attention_maps = calculated_concatenated_attention_maps
    # TODO: I keep it only for the self (=processor) and not for the pipe
    self.pipe.concatenated_attention_maps_over_all_steps_and_attention_modules = concatenated_attention_maps_over_all_steps_and_attention_modules
    if normalized_attention_map is not None:
        self.pipe.mirror_attention_map = normalized_attention_map

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