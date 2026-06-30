## Phase IV Lab: Image-to-Patch Tensor Engine

### Goal 

Write an image processing script using python pure pytorch tensor operetions. we will generate a synthetic color image, slice it up into localized, non-overlapping patches, and flatten those patches into visual vector sequences.

This is the exact data pipeline used by modern multimodal models to "see" charts, UI designs, or photos!

### The Blueprint & Instructions

- Create an Image Tensor: Generate a dummy color image with shape (Channels=3, Height=4, Width=4).

- Slice into Patches: Cut this $4\times4$ image into four smaller $2\times2$ local square patches.

- Flatten to Sequences: Flatten each patch into a single 1D vector of numbers. Since each patch is $3\times2\times2$, your vector length per patch will be exactly 12.

- Stack the Results: Arrange the patches sequentially. Your final output shape must be (Sequence Length, Patch Features).

### Harness & Expected Results

```
import torch

# --- DO NOT ALTER THIS DATA SETUP ---
torch.manual_seed(42)

# Simulating a small 3-channel (RGB) image of size 4x4 pixels
# Shape: (Channels, Height, Width)
mock_image = torch.randint(0, 255, (3, 4, 4)).float()

# --- TODO: YOUR IMPLEMENTATION HERE ---
# Task: Slice 'mock_image' into 4 non-overlapping patches of size 2x2.
# Then flatten each patch across its channels and pixels into a single vector.
# Combine them into a single tensor named 'image_patches'.

image_patches = torch.zeros((4, 12)) # Replace this line with your slicing/flattening logic


# --- VERIFICATION SYSTEM ---
print("--- PHASE IV COMPUTER VISION VERIFICATION ---")
print(f"Original Image Shape (C, H, W): {mock_image.shape}")
print(f"Processed Patch Tensor Shape:    {image_patches.shape}")

# Safety check assertions
assert image_patches.shape == (4, 12), f"Expected shape (4, 12), got {image_patches.shape}"
print("\nSuccess! Your visual tensor pipeline matches the multimodal blueprint.")
```

### Expected Outputs to Match:

- Original Image Shape: torch.Size([3, 4, 4])

- Processed Patch Tensor Shape: torch.Size([4, 12]) (Representing a sequence of 4 visual inputs, where each input contains 12 raw color feature values).