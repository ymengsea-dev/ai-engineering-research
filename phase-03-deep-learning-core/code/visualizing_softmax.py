import numpy as np

def softmax(logits):
    # Step 1: Exponentiate the scores
    exp_scores = np.exp(logits)
    
    # Step 2: Sum all the exponentiated scores
    sum_exp_scores = np.sum(exp_scores)
    
    # Step 3: Divide to get probabilities
    probabilities = exp_scores / sum_exp_scores
    return probabilities

# Target vocabulary: ["code", "banana", "river"]
# These are the raw outputs (logits) from our AI model's final layer
sample_logits = np.array([4.2, -1.5, 1.1])

probs = softmax(sample_logits)

print("--- SOFTMAX VERIFICATION ---")
print(f"Raw Logits:    {sample_logits}")
print(f"Probabilities: {probs}")
print(f"Sum of Probs:  {np.sum(probs):.1f} (Must equal 1.0)")
print("\nInterpretation:")
print(f"There is a {probs[0]*100:.2f}% chance the next word is 'code'.")
print(f"There is a {probs[1]*100:.2f}% chance the next word is 'banana'.")
print(f"There is a {probs[2]*100:.2f}% chance the next word is 'river'.")