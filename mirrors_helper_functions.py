# TODO: Object oriented? maybe get all of them into one class?
# TODO: Reorder imports
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
from diffusers import StableDiffusionXLPipeline
from diffusers.models.attention import Attention
from scipy.ndimage import binary_dilation
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import PIL
import random

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
                                        display_option: str = None,
                                        module_to_display: str = None
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
    
    # TODO: Remove line below if not necessary
    normalized_attention_map = None
    
    if concatenated_current_module_attention_maps.shape[0] == 50:
        # TODO: Remove line below?
        concatenated_attention_maps_over_all_steps_and_attention_modules = None
        if display_option == "Display attention map per layer":
            display_attention_maps_per_layer(concatenated_attention_maps=concatenated_current_module_attention_maps, 
                                tokenized_prompt=tokenized_prompt, 
                                module_name=module_name,
                                average_flag=False
                                )
        if module_name == module_to_display:
            if display_option == "Display average attention map of given module over timestep range":
                    normalized_attention_map = get_avg_attention_map_of_given_module_over_timestep_range(concatenated_attention_maps=concatenated_current_module_attention_maps, 
                                        tokenized_prompt=tokenized_prompt, 
                                        module_name=module_name,
                                        timestep_range=(11, 16)
                                        )
            if display_option == "Display attention map of given module over all timesteps":
                    display_attention_map_of_given_module_over_all_timesteps(concatenated_attention_maps=concatenated_current_module_attention_maps, 
                                    tokenized_prompt=tokenized_prompt, 
                                    module_name=module_name,
                                    )
        # display_thresholded_attention_map_of_given_module(concatenated_attention_maps=concatenated_current_module_attention_maps, 
        #                       tokenized_prompt=tokenized_prompt, 
        #                       module_name=module_name
        # )
        
        # display_edges_using_canny_edge_detector(concatenated_attention_maps=concatenated_current_module_attention_maps, 
        #                       tokenized_prompt=tokenized_prompt, 
        #                       module_name=module_name)
    #   concatenated_attention_maps_over_all_steps_and_attention_modules = concatenate_current_module_attention_maps_to_all_attention_maps(concatenated_current_module_attention_maps, 
                                                                    #   concatenated_attention_maps_over_all_steps_and_attention_modules)
    output = torch.matmul(attention_weights, value)
    return output, concatenated_current_module_attention_maps, concatenated_attention_maps_over_all_steps_and_attention_modules, normalized_attention_map


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
    else:
      average_concatenated_attention_maps_over_all_timesteps = concatenated_attention_maps[-5,:,:]
    
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


def display_attention_map_of_given_module_over_all_timesteps(
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
  thresholds = ["Unthresholded", "Mean", "Median", "75th Percentile"]

  fig, axes = plt.subplots(4, len(tokens), figsize=(num_images * 2, 10))
  fig.suptitle(f"Attention maps for module {module_name}", fontsize=16, y=1.02)

  for i, token in enumerate(tokens):
      attention_map_tensor = concatenated_attention_maps[-1, :, i]
      mean_value = np.mean(attention_map_tensor)
      median_value = np.median(attention_map_tensor)
      percentile_75 = np.percentile(attention_map_tensor, 75)

      # Unthresholded
      unthresholded_map = attention_map_tensor.reshape(32, 32)
      axes[0, i].imshow(unthresholded_map, cmap='viridis')
      axes[0, i].set_title(f"{token}\n{thresholds[0]}")
      axes[0, i].axis("off")

      # Thresholded by mean
      mean_thresholded_map = np.where(attention_map_tensor > mean_value, 1, 0).reshape(32, 32)
      axes[1, i].imshow(mean_thresholded_map, cmap='viridis')
      axes[1, i].set_title(thresholds[1])
      axes[1, i].axis("off")

      # Thresholded by median
      median_thresholded_map = np.where(attention_map_tensor > median_value, 1, 0).reshape(32, 32)
      axes[2, i].imshow(median_thresholded_map, cmap='viridis')
      axes[2, i].set_title(thresholds[2])
      axes[2, i].axis("off")

      # Thresholded by 75th percentile
      percentile_75_thresholded_map = np.where(attention_map_tensor > percentile_75, 1, 0).reshape(32, 32)
      axes[3, i].imshow(percentile_75_thresholded_map, cmap='viridis')
      axes[3, i].set_title(thresholds[3])
      axes[3, i].axis("off")

  plt.tight_layout(rect=[0, 0, 1, 0.96])
  plt.show()


def display_edges_using_canny_edge_detector(
    concatenated_attention_maps: np.ndarray,
    tokenized_prompt: list = None,
    module_name: str = ""
) -> None:
    tokens = ["start"] + tokenized_prompt + ["end"]
    
    mirror_token_index = tokens.index("mirror</w>") if "mirror</w>" in tokens else -1
    if mirror_token_index == -1:
        print("Token 'mirror</w>' not found in tokens list.")
        return

    print(f"Inside 'display_edges_using_canny_edge_detector', concatenated_attention_maps shape is: {concatenated_attention_maps.shape}")
    attention_map_tensor = concatenated_attention_maps[-1, :, mirror_token_index].reshape(32, 32).astype(np.float32)
    upsampled_attention_map_tensor = cv2.resize(attention_map_tensor, (1024, 1024))
    
    scaled_attention_map_ndarray = (upsampled_attention_map_tensor * 255).astype(np.uint8)
    mean_value = np.mean(scaled_attention_map_ndarray)
    median_value = np.median(scaled_attention_map_ndarray)
    percentile_75 = np.percentile(scaled_attention_map_ndarray, 75)
    
    lower = int(max(0, 0.67 * median_value))
    upper = int(min(255, median_value))
    
    edges = cv2.Canny(scaled_attention_map_ndarray, lower, upper)
    
    # Display the attention map and edges side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(upsampled_attention_map_tensor, cmap='viridis')
    axes[0].set_title("Attention Map")
    
    axes[1].imshow(edges, cmap='gray')
    axes[1].set_title(f"Canny Edges")
    
    plt.show()

def get_avg_attention_map_of_given_module_over_timestep_range(
    concatenated_attention_maps: np.ndarray,
    tokenized_prompt: list = None,
    module_name: str = "",
    timestep_range: tuple = (11, 16)
    ) -> np.ndarray:
    tokens = ["start"] + tokenized_prompt + ["end"]
    mirror_token_index = tokens.index("mirror</w>") if "mirror</w>" in tokens else -1
    if mirror_token_index == -1:
        print("Token 'mirror</w>' not found in tokens list.")
        return
    
    attention_map = np.mean(
    np.concatenate([
        concatenated_attention_maps[timestep_range[0]:timestep_range[1], :, mirror_token_index],
        concatenated_attention_maps[-3:, :, mirror_token_index]
    ], axis=0),
    axis=0
    ).reshape(32, 32).astype(np.float32)
    
    min_value = np.min(attention_map)
    max_value = np.max(attention_map)
    normalized_upsampled_attention_map = (attention_map - min_value) / (max_value - min_value)
    
    return normalized_upsampled_attention_map

def preprocess_and_upsample_attention_map(kernel_size: tuple=(2, 2), attention_map: np.ndarray=None, image_target_size: tuple=(1024, 1024)) -> np.ndarray:
    kernel = np.ones(kernel_size, np.uint8)
    dilated_attention_map = cv2.dilate(attention_map, kernel, iterations=1)
    closed_attention_map = cv2.morphologyEx(dilated_attention_map, cv2.MORPH_CLOSE, kernel)
    upsampled_attention_map = cv2.resize(closed_attention_map, image_target_size)
    return upsampled_attention_map


def display_images(images, titles, figsize=(24, 5), max_images_per_row=3):
    num_images = len(images)
    num_rows = (num_images + max_images_per_row - 1) // max_images_per_row  # Calculate total rows needed
    fig, axes = plt.subplots(num_rows, max_images_per_row, figsize=(5 * max_images_per_row, 5 * num_rows))

    # Flatten axes array for easy indexing, in case of fewer than max_images_per_row images in the last row
    axes = axes.ravel() if num_images > 1 else [axes]

    for i in range(num_images):
        axes[i].imshow(images[i], cmap="gray" if images[i].ndim == 2 else None)
        axes[i].set_title(titles[i])
        axes[i].axis("off")
    
    # Turn off any remaining empty subplots
    for j in range(num_images, len(axes)):
        axes[j].axis("off")

    plt.show()
    
def generate_mirror_mask(image: np.ndarray=None, mirror_attention_map: np.ndarray=None) -> np.ndarray:
    image_np = np.array(image)
    grayscale_image = np.array(image.convert("L"))
    scaled_attention_map = 1 / (1 + np.exp(-5 * (mirror_attention_map - 0.5)))
    
    images = [
        image_np,
        grayscale_image,
        scaled_attention_map
    ]
    titles = [
        "Selected Mask",
        "Grayscale Original Image",
        "Scaled Att Map After Sigmoid",
    ]
    
    percentile_values = [30, 40, 50, 60, 70, 80, 90]
    
    areas_ratios_and_maps = []
    
    resized_attention_map = preprocess_and_upsample_attention_map(attention_map=scaled_attention_map)
    # resized_attention_map = cv2.resize(scaled_attention_map, (1024, 1024))
    for percentile_value in percentile_values:
        threshold_value = np.percentile(resized_attention_map, percentile_value)
        thresholded_map = np.where(resized_attention_map > threshold_value, 1, 0).astype(np.uint8)
        contours, _ = cv2.findContours(thresholded_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            current_image = np.copy(image_np)
            cv2.drawContours(current_image, [largest_contour], -1, (0, 0, 255), thickness=2)
            mask_precision_ratio = calculate_mask_precision_ratio(largest_contour, resized_attention_map)
            images.append(current_image)
            titles.append(f"Smoothed AttMap, Thr per {percentile_value},\nMask Precision Ratio: {mask_precision_ratio}")
            
            
            
            areas_ratios_and_maps.append((mask_precision_ratio, area, current_image, f"Thr {percentile_value},\nMask Precision Ratio: {mask_precision_ratio}\n Area {area}"))
        else:
            images.append(thresholded_map)
            titles.append(f"Thr Att Map by per {percentile_value},\nArea: None")

    areas_ratios_and_maps.sort(key=lambda x: x[0])
    
    sorted_areas = [area for _, area, _, _ in areas_ratios_and_maps]

    plt.figure(figsize=(8, 5))
    plt.plot(sorted_areas, marker='o')
    plt.xlabel("Map Index (Sorted by Area)")
    plt.ylabel("Area")
    plt.title("Sorted Areas of Attention Maps")

    # Add the titles as x-ticks
    titles_for_xticks = [title for _, _, _, title in areas_ratios_and_maps]
    plt.xticks(range(len(titles_for_xticks)), titles_for_xticks, rotation=90, fontsize=8)  # Adjust rotation and font size as needed
    plt.tight_layout()
    plt.show()

    highest_ratio_entry = max(areas_ratios_and_maps, key=lambda x: x[0])
    highest_ratio_image, highest_ratio_title = highest_ratio_entry[2], highest_ratio_entry[3]
    images[0] = highest_ratio_image
    titles[0] = "Selected Mask: " + highest_ratio_title

    display_images(images, titles)
    
    return
        
        
def calculate_mask_precision_ratio(smoothed_contour, attention_map: np.ndarray=None) -> float:
    mask = np.zeros(attention_map.shape, dtype=np.uint8)
    
    cv2.fillPoly(mask, [smoothed_contour], 1)
    
    inside_pixels = attention_map[mask == 1]
    inside_avg = np.mean(inside_pixels) if inside_pixels.size > 0 else 0

    outside_pixels = attention_map[mask == 0]
    outside_avg = np.mean(outside_pixels) if outside_pixels.size > 0 else 0

    precision_ratio = inside_avg / outside_avg if outside_avg != 0 else float('inf')
    
    return precision_ratio

def display_SAM_masks(image: PIL.Image.Image=None, mirror_attention_mask: np.ndarray=None) -> None:
    scaled_attention_map = 1 / (1 + np.exp(-5 * (mirror_attention_mask - 0.5)))
    
    resized_attention_map = preprocess_and_upsample_attention_map(attention_map=scaled_attention_map)
    
    sam_model = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
    mask_generator = SamAutomaticMaskGenerator(sam_model)

    image_np = np.array(image)
    masks = mask_generator.generate(image_np)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(image)
    mask_colors = [(random.random(), random.random(), random.random()) for _ in masks]
    for mask, color in zip(masks, mask_colors):
        mask_array = mask["segmentation"]
        outline = binary_dilation(mask_array) & ~mask_array
        ax1.contour(outline, colors=[color], linewidths=1)
    ax1.axis("off")
    ax1.set_title("SAM Masks on Image")

    ax2.imshow(resized_attention_map, cmap="gray")
    for mask, color in zip(masks, mask_colors):
        mask_array = mask["segmentation"]
        outline = binary_dilation(mask_array) & ~mask_array
        ax2.contour(outline, colors=[color], linewidths=1)
    ax2.axis("off")
    ax2.set_title("SAM Masks on Mirror Attention Mask")

    plt.show()