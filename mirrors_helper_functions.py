# TODO: Reorder imports
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math


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
      display_attention_maps(concatenated_attention_maps=concatenated_attention_maps)

    # 4. Multiply by the values
    output = torch.matmul(attention_weights, value)

    return output, concatenated_attention_maps

def display_attention_maps(concatenated_attention_maps: torch) -> None:
    # average_concatenated_attention_maps_over_all_timesteps = concatenated_attention_maps.mean(axis=0)
    # image_resolution_height_and_width = int(math.sqrt(average_concatenated_attention_maps_over_all_timesteps.shape[0]))
    # # TODO: create if... else... with special exception instead of assert
    # assert image_resolution_height_and_width ** 2 == average_concatenated_attention_maps_over_all_timesteps.shape[0]
    
    # images = [average_concatenated_attention_maps_over_all_timesteps[:, i].reshape(image_resolution_height_and_width, image_resolution_height_and_width) for i in range(77)]
    # fig, axes = plt.subplots(11, 7, figsize=(14, 22))
    # fig.suptitle("Attention Maps")
    # axes = axes.flatten()
    # for i in range(77):
    #     ax = axes[i]
    #     ax.imshow(images[i], cmap='viridis')
    # plt.tight_layout(rect=[0, 0, 0.5, 0.5])
    # plt.show()
    return

