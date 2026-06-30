## Text Preprocessing & Embedding Engine

### The Goal

objective is to build a text processing pipeline using pure Python and PyTorch. You will parse a text corpus, generate a vocabulary dictionary that includes a special `<PAD>` token, tokenize and pad the sequences uniformly, and extract true continuous feature representations from an embedding module.

### The Blueprint & Instructions

1. Build `word2idx`: Extract all unique words from the corpus below, lowercase them, and assign them an index starting from 1. Put a "`<PAD>`" key at index 0.

2. Tokenize and Pad: Convert each sentence in the corpus to lowercased tokens. Substitute those words for their index integer. Pad the shorter sequences with 0 so all sentences have a final length of 4.

3. Verify via PyTorch: Pass your padded integer arrays into a `torch.nn.Embedding` layer and assert that the final numeric block shape matches target shape dimensions perfectly.

### Harness & Expected Results

```
import torch
import torch.nn as nn

# --- DO NOT ALTER THIS DATA SETUP ---
torch.manual_seed(42)

corpus = [
    "I love coding AI",
    "AI can write code",
    "I love deep learning"
]

# 1. TODO: BUILD YOUR VOCABULARY
# Parse the corpus sentences, split them, lowercase them, and populate word2idx.
# Your dictionary MUST start with {"<PAD>": 0}
word2idx = {"<PAD>": 0}

# Add your loop logic here to fill word2idx dynamically...


# 2. TODO: TOKENIZE, NUMERICALIZE, AND PAD
# Process each sentence into a list of 4 integer IDs.
# Shorter lists must fill the empty space with 0.
padded_indices = []

# Add your text splitting and padding logic here...


# --- AUTOMATED PROFESSOR VERIFICATION SYSTEM ---
try:
    # Convert your list of lists into a PyTorch LongTensor
    input_tensor = torch.tensor(padded_indices, dtype=torch.long)

    vocab_size = len(word2idx)
    embedding_dim = 8  # Transform each token into an 8-dimensional vector

    embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
    word_vectors = embedding_layer(input_tensor)

    print("--- PHASE V NLP FOUNDATIONS VERIFICATION ---")
    print(f"Generated Vocabulary: {word2idx}\n")
    print(f"Input Tensor Shape (Batch, Seq Length): {input_tensor.shape}")
    print(f"Output Vector Shape (Batch, Seq Length, Embedding Dim): {word_vectors.shape}")

    # Assertions for correctness
    assert vocab_size == 9, f"Expected 9 unique tokens (8 words + 1 PAD), got {vocab_size}"
    assert input_tensor.shape == (3, 4), f"Expected input tensor shape (3, 4), got {input_tensor.shape}"
    assert word_vectors.shape == (3, 4, 8), f"Expected embedding vector shape (3, 4, 8), got {word_vectors.shape}"

    print("\nSuccess! Your data preprocessing engine is fully certified.")

except Exception as e:
    print(f"\n❌ Verification Failed. Error: {e}")
```

### Expected Target Outputs to Match:

- Vocabulary Size: Exactly 9 items (including your tracking pad entry).

- Input Tensor Shape: Exactly `torch.Size([3, 4])`.

- Output Embedding Vector Shape: Exactly torch.Size(`[3, 4, 8]`).
