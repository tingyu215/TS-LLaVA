#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape

from einops import rearrange
import math


def uniformly_sample_frames(frames, num_samples=4):
    """
    Uniformly samples frames from a batch of frames.

    Args:
    - frames (tensor): A tensor of shape [N, C, H, W].
    - num_samples (int): The number of frames to sample.

    Returns:
    - sampled_frames (tensor): A tensor of sampled frames with shape [num_samples, C, H, W].
    """
    N = frames.shape[0]
    if N < num_samples:
        raise ValueError(f"Cannot sample {num_samples} frames from a tensor with only {N} frames.")

    # Compute indices for uniform sampling
    indices = torch.linspace(0, N - 1, steps=num_samples)
    indices = torch.round(indices).long()

    # Ensure indices are within valid range
    indices = torch.clamp(indices, 0, N - 1)

    # Sample frames
    sampled_frames = frames[indices]
    return sampled_frames


def resize_images(images, size):
    """
    Resizes a batch of images to the specified size using optimal interpolation methods.

    Args:
    - images (tensor): A tensor of shape [N, C, H, W].
    - size (tuple): The desired size (height, width) for the images.

    Returns:
    - resized_images (tensor): A tensor of resized images with shape [N, C, new_H, new_W].
    """
    original_height, original_width = images.shape[2], images.shape[3]
    new_height, new_width = size

    if new_height < original_height or new_width < original_width:
        # Downsampling - use area interpolation
        resized_images = nn.functional.interpolate(
            images, size=size, mode='area'
        )
    else:
        # Upsampling - use bilinear interpolation
        resized_images = nn.functional.interpolate(
            images, size=size, mode='bilinear', align_corners=False
        )
    return resized_images

def find_best_grid(num_images):
    """
    Finds the grid dimensions (rows and columns) that best fit the number of images,
    minimizing the difference between rows and columns.

    Args:
    - num_images (int): The number of images to arrange in the grid.

    Returns:
    - best_rows (int): The optimal number of grid rows.
    - best_cols (int): The optimal number of grid columns.
    """
    factors = []
    for i in range(1, int(math.sqrt(num_images)) + 1):
        if num_images % i == 0:
            factors.append((i, num_images // i))

    # Find the factor pair with minimal difference between rows and columns
    best_rows, best_cols = min(factors, key=lambda x: abs(x[0] - x[1]))

    # Ensure that rows >= cols for consistency (optional)
    if best_rows < best_cols:
        best_rows, best_cols = best_cols, best_rows

    return best_rows, best_cols

def create_image_grid(images, final_height=336, final_width=336, shuffling=False):
    """
    Combines a batch of images into a grid.

    Args:
    - images (tensor): A tensor of shape [N, C, H, W].
    - final_height (int): Desired height of the final grid image.
    - final_width (int): Desired width of the final grid image.
    - shuffling (bool): whether to shuffle the selected frames.

    Returns:
    - grid_image (tensor): Combined image grid tensor with shape [C, final_height, final_width].
    """
    num_images = images.shape[0]
    channels = images.shape[1]

    # Find the best grid dimensions
    grid_rows, grid_cols = find_best_grid(num_images)

    # Compute new size for each image in the grid
    img_height = final_height // grid_rows
    img_width = final_width // grid_cols

    # Resize images using optimal interpolation
    resized_images = resize_images(images, size=(img_height, img_width))  # [N, C, img_height, img_width]

    if shuffling:
        shuffled_indices = torch.randperm(resized_images.size(0))
        resized_images = resized_images[shuffled_indices]

    # Reshape and rearrange images to form the grid
    # Resized images shape: [N, C, img_height, img_width]
    # First, reshape to [grid_rows, grid_cols, C, img_height, img_width]
    grid_images = resized_images.view(grid_rows, grid_cols, channels, img_height, img_width)

    # Permute dimensions to bring channels to the front
    grid_images = grid_images.permute(2, 0, 3, 1, 4)  # [C, grid_rows, img_height, grid_cols, img_width]

    # Reshape to combine rows and columns
    grid_image = grid_images.contiguous().view(
        channels,
        grid_rows * img_height,
        grid_cols * img_width
    )

    # Ensure the final image has the exact desired dimensions
    grid_image = grid_image[:, :final_height, :final_width]

    return grid_image  # Shape: [C, final_height, final_width]



def sample_tokens(tensor, num_samples=1728):
    """
    Uniformly samples tokens from a tensor, including both the first and last tokens.
    
    Args:
    - tensor (torch.Tensor): A tensor of shape [N, D], where N is the total number of tokens.
    - num_samples (int): The number of tokens to sample.
    
    Returns:
    - sampled_tokens (torch.Tensor): A tensor of shape [num_samples, D].
    """
    N = tensor.shape[0]
    if N < num_samples:
        raise ValueError(f"Cannot sample {num_samples} tokens from a tensor with only {N} tokens.")

    # Compute indices using linspace to include the first and last indices
    indices = torch.linspace(0, N - 1, steps=num_samples)
    indices = torch.round(indices).long()
    indices = torch.clamp(indices, 0, N - 1)  # Ensure indices are within valid range

    # Sample tokens
    sampled_tokens = tensor[indices]
    return sampled_tokens


def make_multiple_grids(frames, chunk_size=4):
    images = []
    num_chunks = frames.size(0) // chunk_size  # Adjusted to accommodate the new chunk size

    for i in range(num_chunks):
        # Extract a tensor of shape [chunk_size,1,3,336,336]
        start_idx = i * chunk_size
        frames_part = frames[start_idx:start_idx + chunk_size, 0, :, :, :]  # Adjusted the slice to use chunk_size
    
        # Pass frames_part through your create_image_grid() function
        img = create_image_grid(frames_part)  # Outputs a tensor of shape [1,3,336,336]

        images.append(img)
    return images


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features


    def grid_aggregation(self, images):
        if images.ndim==4:
            images = images.unsqueeze(1)
        if self.config.aggregation_method.startswith("X"):
            if self.config.aggregation_method == "X1":
                sampled_images = uniformly_sample_frames(images[:, 0, :, :, :].squeeze(), num_samples=4)
            elif self.config.aggregation_method == "X2":
                sampled_images = uniformly_sample_frames(images[:, 0, :, :, :].squeeze(), num_samples=6)
            elif self.config.aggregation_method == "X3":
                sampled_images = uniformly_sample_frames(images[:, 0, :, :, :].squeeze(), num_samples=8)
            else:
                raise NotImplementedError
            sampled_images = create_image_grid(sampled_images)
            image_features = self.encode_images(sampled_images.unsqueeze(0))
            # image_features = image_features.unsqueeze(0)
        elif self.config.aggregation_method.startswith("Y"):
            if self.config.aggregation_method.startswith("Y1"):
                sampled_images = uniformly_sample_frames(images[:, 0, :, :, :].squeeze(), num_samples=4)
            elif self.config.aggregation_method.startswith("Y2"):
                sampled_images = uniformly_sample_frames(images[:, 0, :, :, :].squeeze(), num_samples=6)
            elif self.config.aggregation_method.startswith("Y3"):
                sampled_images = uniformly_sample_frames(images[:, 0, :, :, :].squeeze(), num_samples=8)
            else:
                raise NotImplementedError
            
            sampled_images = create_image_grid(sampled_images)
            grid_image_features = self.encode_images(sampled_images.unsqueeze(0))

            # only use base image feature
            sampled_image_features = self.encode_images(images[:, 0, :, :, :].squeeze())
            sampled_image_features = sampled_image_features.view(-1, sampled_image_features.shape[-1])

            # sample tokens from features generating from all frames
            # self.config.num_sampled_tokens + tokens from thumbnail image < LLM context length
            sampled_image_features = sample_tokens(sampled_image_features, num_samples=self.config.num_sampled_tokens)

                
            sampled_image_features = sampled_image_features.unsqueeze(0)
            image_features = torch.cat((grid_image_features, sampled_image_features), dim=1)
        elif self.config.aggregation_method.startswith("Z"):
            if self.config.aggregation_method == "Z1":
                chunk_size = 4
            elif self.config.aggregation_method == "Z2":
                chunk_size = 6
            elif self.config.aggregation_method == "Z3":
                chunk_size = 8
            else:
                raise NotImplementedError

            grid_image_list = make_multiple_grids(images, chunk_size=chunk_size)
            grid_images = torch.stack(grid_image_list, dim=0)
            image_features = self.encode_images(grid_images)
            image_features = image_features.view(-1, image_features.shape[-1])
            image_features = image_features.unsqueeze(0)
        elif self.config.aggregation_method.startswith("W"):
            if self.config.aggregation_method.startswith("W1"):
                # 2 times 4grid
                sampled_images = uniformly_sample_frames(images[:, 0, :, :, :].squeeze(), num_samples=8)
                grid_image_list = make_multiple_grids(sampled_images.unsqueeze(1), chunk_size=4)

                grid_images = torch.stack(grid_image_list, dim=0)
                grid_image_features = self.encode_images(grid_images)
                grid_image_features = grid_image_features.view(-1, grid_image_features.shape[-1])
                grid_image_features = grid_image_features.unsqueeze(0)
            elif self.config.aggregation_method.startswith("W2"):
                # 3 times 4grid
                sampled_images = uniformly_sample_frames(images[:, 0, :, :, :].squeeze(), num_samples=12)
                grid_image_list = make_multiple_grids(sampled_images.unsqueeze(1), chunk_size=4)

                grid_images = torch.stack(grid_image_list, dim=0)
                grid_image_features = self.encode_images(grid_images)
                grid_image_features = grid_image_features.view(-1, grid_image_features.shape[-1])
                grid_image_features = grid_image_features.unsqueeze(0)
            elif self.config.aggregation_method.startswith("W3"):
                # 2 times 6grid
                sampled_images = uniformly_sample_frames(images[:, 0, :, :, :].squeeze(), num_samples=12)
                grid_image_list = make_multiple_grids(sampled_images.unsqueeze(1), chunk_size=6)

                grid_images = torch.stack(grid_image_list, dim=0)
                grid_image_features = self.encode_images(grid_images)
                grid_image_features = grid_image_features.view(-1, grid_image_features.shape[-1])
                grid_image_features = grid_image_features.unsqueeze(0)
            elif self.config.aggregation_method.startswith("W4"):
                # 3 times 6grid
                sampled_images = uniformly_sample_frames(images[:, 0, :, :, :].squeeze(), num_samples=18)
                grid_image_list = make_multiple_grids(sampled_images.unsqueeze(1), chunk_size=6)

                grid_images = torch.stack(grid_image_list, dim=0)
                grid_image_features = self.encode_images(grid_images)
                grid_image_features = grid_image_features.view(-1, grid_image_features.shape[-1])
                grid_image_features = grid_image_features.unsqueeze(0)
            else:
                raise NotImplementedError

            # only use base image feature
            sampled_image_features = self.encode_images(images[:, 0, :, :, :].squeeze())
            sampled_image_features = sampled_image_features.view(-1, sampled_image_features.shape[-1])

            if self.config.aggregation_method.endswith("a"):
                sampled_image_features = sample_tokens(sampled_image_features, num_samples=1728) # with 3 grids,adds up to 3456 tokens (6*576)
            elif self.config.aggregation_method.endswith("b"):
                sampled_image_features = sample_tokens(sampled_image_features, num_samples=2304) # with 2 grids, adds up to 3456 tokens (6*576)
            else:
                raise NotImplementedError
            
            sampled_image_features = sampled_image_features.unsqueeze(0)
            image_features = torch.cat((grid_image_features, sampled_image_features), dim=1)
        
        elif self.config.aggregation_method.startswith("V"):
            if self.config.aggregation_method.startswith("V1"):
                sampled_images = uniformly_sample_frames(images[:, 0, :, :, :].squeeze(), num_samples=4)
            elif self.config.aggregation_method.startswith("V2"):
                sampled_images = uniformly_sample_frames(images[:, 0, :, :, :].squeeze(), num_samples=6)
            elif self.config.aggregation_method.startswith("V3"):
                sampled_images = uniformly_sample_frames(images[:, 0, :, :, :].squeeze(), num_samples=8)
            else:
                raise NotImplementedError
            
            if self.config.aggregation_method.endswith("z"):
                do_shuffle=True
            else:
                do_shuffle=False
            sampled_images = create_image_grid(sampled_images, shuffling=do_shuffle)
            grid_image_features = self.encode_images(sampled_images.unsqueeze(0))

            # only use base image feature
            sampled_image_features = self.encode_images(images[:, 0, :, :, :].squeeze())
            sampled_image_features = sampled_image_features.view(-1, sampled_image_features.shape[-1])

            # sample tokens from features generating from all frames
            # self.config.num_sampled_tokens + tokens from thumbnail image < LLM context length
            sampled_image_features = sample_tokens(sampled_image_features, num_samples=self.config.num_sampled_tokens)

                
            sampled_image_features = sampled_image_features.unsqueeze(0)
            image_features = torch.cat((sampled_image_features, grid_image_features), dim=1)
        elif self.config.aggregation_method.startswith("U"):
            if self.config.aggregation_method.startswith("U1"):
                # 2 times 4grid
                sampled_images = uniformly_sample_frames(images[:, 0, :, :, :].squeeze(), num_samples=8)
                grid_image_list = make_multiple_grids(sampled_images.unsqueeze(1), chunk_size=4)

                grid_images = torch.stack(grid_image_list, dim=0)
                grid_image_features = self.encode_images(grid_images)
                grid_image_features = grid_image_features.view(-1, grid_image_features.shape[-1])
                grid_image_features = grid_image_features.unsqueeze(0)
            elif self.config.aggregation_method.startswith("U2"):
                # 3 times 4grid
                sampled_images = uniformly_sample_frames(images[:, 0, :, :, :].squeeze(), num_samples=12)
                grid_image_list = make_multiple_grids(sampled_images.unsqueeze(1), chunk_size=4)

                grid_images = torch.stack(grid_image_list, dim=0)
                grid_image_features = self.encode_images(grid_images)
                grid_image_features = grid_image_features.view(-1, grid_image_features.shape[-1])
                grid_image_features = grid_image_features.unsqueeze(0)
            elif self.config.aggregation_method.startswith("U3"):
                # 2 times 6grid
                sampled_images = uniformly_sample_frames(images[:, 0, :, :, :].squeeze(), num_samples=12)
                grid_image_list = make_multiple_grids(sampled_images.unsqueeze(1), chunk_size=6)

                grid_images = torch.stack(grid_image_list, dim=0)
                grid_image_features = self.encode_images(grid_images)
                grid_image_features = grid_image_features.view(-1, grid_image_features.shape[-1])
                grid_image_features = grid_image_features.unsqueeze(0)
            elif self.config.aggregation_method.startswith("U4"):
                # 3 times 6grid
                sampled_images = uniformly_sample_frames(images[:, 0, :, :, :].squeeze(), num_samples=18)
                grid_image_list = make_multiple_grids(sampled_images.unsqueeze(1), chunk_size=6)

                grid_images = torch.stack(grid_image_list, dim=0)
                grid_image_features = self.encode_images(grid_images)
                grid_image_features = grid_image_features.view(-1, grid_image_features.shape[-1])
                grid_image_features = grid_image_features.unsqueeze(0)
            else:
                raise NotImplementedError

            # only use base image feature
            sampled_image_features = self.encode_images(images[:, 0, :, :, :].squeeze())
            sampled_image_features = sampled_image_features.view(-1, sampled_image_features.shape[-1])

            # sample tokens from features generating from all frames
            # self.config.num_sampled_tokens + tokens from thumbnail image < LLM context length
            sampled_image_features = sample_tokens(sampled_image_features, num_samples=self.config.num_sampled_tokens)
            
            sampled_image_features = sampled_image_features.unsqueeze(0)
            image_features = torch.cat((sampled_image_features, grid_image_features), dim=1)
        else:
            raise NotImplementedError
        return image_features


    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            # images: [T S C H W]
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            # import ipdb
            # ipdb.set_trace()
            if self.config.aggregation_method.startswith("X") or self.config.aggregation_method.startswith("Y") or self.config.aggregation_method.startswith("Z") or self.config.aggregation_method.startswith("W") or self.config.aggregation_method.startswith("V") or self.config.aggregation_method.startswith("U"):
                image_features = self.grid_aggregation(images)
            else:
                concat_images = torch.cat([image for image in images], dim=0)  # [TS C H W] 
                image_features = self.encode_images(concat_images)  # [TS N D]
                split_sizes = [image.shape[0] for image in images]  # T * [S]
                image_features = torch.split(image_features, split_sizes, dim=0)   # T * [S N D]
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                if self.config.aggregation_method.startswith("X") or self.config.aggregation_method.startswith("Y") or self.config.aggregation_method.startswith("Z") or self.config.aggregation_method.startswith("W")  or self.config.aggregation_method.startswith("V") or self.config.aggregation_method.startswith("U"):
                    image_features = image_features
                else:
                    new_image_features = []
                    for image_idx, image_feature in enumerate(image_features):
                        if image_feature.shape[0] > 1:
                            # ipdb.set_trace()
                            base_image_feature = image_feature[0]   # [N D]
                            image_feature = image_feature[1:]  # [S-1 N D]
                            
                            height = width = self.get_vision_tower().num_patches_per_side
                            assert height * width == base_image_feature.shape[0]
                            if image_aspect_ratio == 'anyres':
                                # ipdb.set_trace()
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)  # [sqrt(S-1) sqrt(S-1) H W D]
                            else:
                                raise NotImplementedError
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                                ), dim=-1)
                                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                            else:
                                image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                                image_feature = image_feature.flatten(0, 3)
                            # image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                            # Using base feature only for LLaVA-1.6 to avoid extra tokens from high-resolution features.
                            image_feature = base_image_feature  # [576 D]
                        else:
                            image_feature = image_feature[0]
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ), dim=0)
                        new_image_features.append(image_feature)
                    image_features = new_image_features
                    # len=T, [1948 D]
                    image_features = torch.stack(image_features, dim=0)  # [T New_N D]
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        elif self.config.aggregation_method.startswith("X") or self.config.aggregation_method.startswith("Y") or self.config.aggregation_method.startswith("Z") or self.config.aggregation_method.startswith("W") or self.config.aggregation_method.startswith("V") or self.config.aggregation_method.startswith("U"):
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            image_features = self.grid_aggregation(images)
        else:
            image_features = self.encode_images(images).to(self.device)  # [T 576 D]
        
        if image_aspect_ratio == 'anyres':
            T, N, D = image_features.shape
            image_features = image_features.view(T * N, D)
            image_features = image_features.unsqueeze(0) 
        elif image_aspect_ratio == "resize":
            image_features = image_features
        else:
            raise NotImplementedError
        

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # always 1
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)  # 627: 576 + 51, 1999: 1948 + 51, 51 is prefix
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
