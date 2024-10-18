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
def register_tokenized_prompt_and_module_name_into_pipe_attention_modules(pipe: StableDiffusionXLPipeline = None, tokenized_prompt: list = None) -> None:
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
                                        concatenated_attention_maps=None,
                                        tokenized_prompt: list = None,
                                        module_name: str = ""
                                        ):
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
      display_attention_maps(concatenated_attention_maps=concatenated_attention_maps, 
                             tokenized_prompt=tokenized_prompt, 
                             module_name=module_name
                             )

    # 4. Multiply by the values
    output = torch.matmul(attention_weights, value)

    return output, concatenated_attention_maps


def display_attention_maps(concatenated_attention_maps: torch.Tensor, 
                           tokenized_prompt: list = None,
                           module_name: str = ""
                           ) -> None:
    # Take the mean over all timesteps
    average_concatenated_attention_maps_over_all_timesteps = concatenated_attention_maps.mean(axis=0)
    
    # Calculate image resolution (assuming square images)
    image_resolution_height_and_width = int(math.sqrt(average_concatenated_attention_maps_over_all_timesteps.shape[0]))
    
    # Replace the assert with a proper if-else condition and raise an exception if dimensions don't match
    if image_resolution_height_and_width ** 2 != average_concatenated_attention_maps_over_all_timesteps.shape[0]:
        raise ValueError("Attention map dimensions do not match a square image resolution.")
    
    # Determine how many images to show
    num_images = len(tokenized_prompt) + 2  # +2 for 'start' and 'end'
    
    # Create the images for the attention maps (only the required number of images)
    images = [average_concatenated_attention_maps_over_all_timesteps[:, i].reshape(image_resolution_height_and_width, image_resolution_height_and_width) for i in range(num_images)]
    
    # Set up the figure for displaying images
    
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, math.ceil(num_images / 10) * 2 + 1))
    fig.suptitle(f"Cross Attention Maps for module: {module_name}")
    axes = axes.flatten()

    # Titles for the images
    titles = ['start'] + tokenized_prompt + ['end']  # Create titles for 'start', tokenized_prompt words, and 'end'

    # Display the images with titles
    for i in range(num_images):
        ax = axes[i]
        ax.imshow(images[i], cmap='viridis')
        ax.set_title(titles[i], fontsize=8)  # Assign the corresponding title
    
    # Hide any remaining empty subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
