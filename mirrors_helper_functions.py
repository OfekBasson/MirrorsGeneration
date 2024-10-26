# TODO: Object oriented? maybe get all of them into one class?
# TODO: Reorder imports
# TODO: Auto formatting
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
from diffusers import StableDiffusionXLPipeline
from diffusers.models.attention import Attention

# TODO: Should I transform the "list" into "Optional[List] or something? what are the differences between both?"
def register_tokenized_prompt_and_module_name_and_index_into_pipe_and_its_attention_modules(pipe: StableDiffusionXLPipeline = None, tokenized_prompt: list = None) -> None:
    pipe.tokenized_prompt = tokenized_prompt
    for name, module in pipe.unet.named_modules():
      if name.endswith("attn2") and isinstance(module, Attention):
        module.tokenized_prompt = tokenized_prompt
        module.name = name


def custom_scaled_dot_product_attention(query, 
                                        key, 
                                        value, 
                                        attn_mask=None, 
                                        dropout_p=0.0, 
                                        is_causal=False, 
                                        concatenated_current_module_attention_maps=None,
                                        tokenized_prompt: list = None,
                                        module_name: str = "",
                                        concatenated_attention_maps_over_all_steps_and_attention_modules: torch.Tensor = None,
                                        ):
    d_k = query.size(-1)  
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    if attn_mask is not None:
        scores = scores.masked_fill(attn_mask == 0, float('-inf'))

    attention_weights = F.softmax(scores, dim=-1)
    current_step_attention_map_mean_over_heads = np.expand_dims(attention_weights[1].mean(dim=0).cpu().numpy(), axis=0)
    if concatenated_current_module_attention_maps is None:
      concatenated_current_module_attention_maps = current_step_attention_map_mean_over_heads
    else:
      concatenated_current_module_attention_maps = np.concatenate((concatenated_current_module_attention_maps, current_step_attention_map_mean_over_heads), axis=0)

    if concatenated_current_module_attention_maps.shape[0] == 50:
      if module_name == "up_blocks.0.attentions.0.transformer_blocks.2.attn2" or module_name == "up_blocks.0.attentions.1.transformer_blocks.1.attn2" or module_name == "up_blocks.0.attentions.1.transformer_blocks.7.attn2" or module_name == "up_blocks.0.attentions.1.transformer_blocks.2.attn2":
        # display_last_attention_map_of_given_module_over_all_timesteps(concatenated_attention_maps=concatenated_current_module_attention_maps, 
        #                       tokenized_prompt=tokenized_prompt, 
        #                       module_name=module_name,
        #                       )
        display_thresholded_attention_map_of_given_module(concatenated_attention_maps=concatenated_current_module_attention_maps, 
                              tokenized_prompt=tokenized_prompt, 
                              module_name=module_name
        )
        # display_attention_maps_per_layer(concatenated_attention_maps=concatenated_current_module_attention_maps, 
        #                       tokenized_prompt=tokenized_prompt, 
        #                       module_name=module_name,
        #                       average_flag=False
        #                       )
      concatenated_attention_maps_over_all_steps_and_attention_modules = concatenate_current_module_attention_maps_to_all_attention_maps(concatenated_current_module_attention_maps, 
                                                                      concatenated_attention_maps_over_all_steps_and_attention_modules)
    output = torch.matmul(attention_weights, value)
    return output, concatenated_current_module_attention_maps, concatenated_attention_maps_over_all_steps_and_attention_modules


def concatenate_current_module_attention_maps_to_all_attention_maps(concatenated_current_module_attention_maps: torch.Tensor = None, 
                                                                  concatenated_attention_maps_over_all_steps_and_attention_modules: torch.Tensor = None
                                                                  ) -> torch.Tensor:
    concatenated_current_module_attention_maps = upsample_concatenated_current_module_attention_maps_to_attention_maps_desired_shape(torch.from_numpy(concatenated_current_module_attention_maps))
    if concatenated_attention_maps_over_all_steps_and_attention_modules is None:
        concatenated_attention_maps_over_all_steps_and_attention_modules = concatenated_current_module_attention_maps
    else:
        concatenated_attention_maps_over_all_steps_and_attention_modules = torch.cat(
            (concatenated_attention_maps_over_all_steps_and_attention_modules, concatenated_current_module_attention_maps), dim=0
        )
    
    return concatenated_attention_maps_over_all_steps_and_attention_modules
  
  
def upsample_concatenated_current_module_attention_maps_to_attention_maps_desired_shape(concatenated_current_module_attention_maps: torch.Tensor = None,
                                                                                        attention_maps_desired_shape: int = 4096
                                                                                        ) -> torch.Tensor:
  if concatenated_current_module_attention_maps.shape[1] == attention_maps_desired_shape:
    return concatenated_current_module_attention_maps
  concatenated_current_module_attention_maps = concatenated_current_module_attention_maps.permute(0, 2, 1)
  upscaled_concatenated_current_module_attention_maps = F.interpolate(concatenated_current_module_attention_maps, size=attention_maps_desired_shape, mode='linear', align_corners=False)
  upscaled_concatenated_current_module_attention_maps = upscaled_concatenated_current_module_attention_maps.permute(0, 2, 1)
  return upscaled_concatenated_current_module_attention_maps


def display_attention_maps_per_layer(concatenated_attention_maps: np.array, 
                           tokenized_prompt: list = None,
                           module_name: str = "",
                           average_flag: bool = False
                           ) -> None:
    if average_flag:
      average_concatenated_attention_maps_over_all_timesteps = concatenated_attention_maps.mean(axis=0)
      print("average flag is on")
    else:
      average_concatenated_attention_maps_over_all_timesteps = concatenated_attention_maps[-5,:,:]
      print("average flag is off")
    
    image_resolution_height_and_width = int(math.sqrt(average_concatenated_attention_maps_over_all_timesteps.shape[0]))
    
    if image_resolution_height_and_width ** 2 != average_concatenated_attention_maps_over_all_timesteps.shape[0]:
        raise ValueError("Attention map dimensions do not match a square image resolution.")
    
    num_images = len(tokenized_prompt) + 2  
    images = [average_concatenated_attention_maps_over_all_timesteps[:, i].reshape(image_resolution_height_and_width, image_resolution_height_and_width) for i in range(num_images)]
    
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, math.ceil(num_images / 10) * 2 + 1))
    title_str = "Average Attention Maps Over All Steps And Attention Modules" if module_name == "Average Attention Maps Over All Steps And Attention Modules" else f"Cross Attention Maps for module: {module_name}"
    fig.suptitle(title_str)
    axes = axes.flatten()

    titles = ['start'] + tokenized_prompt + ['end'] 

    for i in range(num_images):
        ax = axes[i]
        ax.imshow(images[i], cmap='viridis')
        ax.set_title(titles[i], fontsize=8)  
    
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def display_last_attention_map_of_given_module_over_all_timesteps(
    concatenated_attention_maps: np.ndarray,
    tokenized_prompt: list = None,
    module_name: str = ""
) -> None:
    tokens = ["start"] + tokenized_prompt + ["end"]
    num_images = len(tokens)
    num_timesteps, _, _ = concatenated_attention_maps.shape

    for t in range(num_timesteps):
        fig, axes = plt.subplots(1, len(tokens), figsize=(num_images * 2, math.ceil(num_images / 10) * 2 + 1))
        fig.suptitle(f"Attention maps for layer {t+1} of module {module_name}", fontsize=16)

        for i, token in enumerate(tokens):
            ax = axes[i]
            attention_map_tensor = concatenated_attention_maps[t, :, i]

            attention_map = attention_map_tensor.reshape(32, 32)

            ax.imshow(attention_map, cmap='viridis')
            ax.set_title(token)
            ax.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    
def display_thresholded_attention_map_of_given_module(
    concatenated_attention_maps: np.ndarray,
    tokenized_prompt: list = None,
    module_name: str = ""
) -> None:
  tokens = ["start"] + tokenized_prompt + ["end"]
  num_images = len(tokens)

  fig, axes = plt.subplots(1, len(tokens), figsize=(num_images * 2, math.ceil(num_images / 10) * 2 + 1))
  fig.suptitle(f"Attention maps for module {module_name}", fontsize=16)

  for i, token in enumerate(tokens):
      ax = axes[i]
      attention_map_tensor = concatenated_attention_maps[-1, :, i]
      mean_value = attention_map_tensor.median()
      thresholded_attention_map = np.where(attention_map_tensor > mean_value, 1, 0).reshape(32, 32)

      ax.imshow(thresholded_attention_map, cmap='viridis')
      ax.set_title(token)
      ax.axis("off")

  plt.tight_layout(rect=[0, 0, 1, 0.96])
  plt.show()
