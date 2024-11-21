# %%
import torch
import torch.nn.functional as F
relu = torch.nn.ReLU()

def relu_l1_norm(logits):
    relu_output = F.relu(logits)  # Apply ReLU
    l1_norm = torch.linalg.vector_norm(relu_output, ord=1, dim=1, keepdim=True)  # Compute L1 norm along C
    l1_norm = torch.where(l1_norm == 0, torch.ones_like(l1_norm), l1_norm)  # Avoid division by zero
    return relu_output / l1_norm

def temperature_scaled_softmax(tensor, dim:int = 0, temperature=1.0):
    tensor = tensor / temperature
    return torch.softmax(tensor, dim=dim)
# %%
B, C, H, W, D = 1, 4, 1, 1, 2
logit = torch.randn(B, C, H, W, D)
print(logit)
# %%
if bool(relu(logit).max()): # checking if the max value of the relu is not zero
    probs = relu(logit)/relu(logit).max()
else: 
    probs = relu(logit)

print(probs)
# %%
print(relu_l1_norm(logit))

# %%
print(temperature_scaled_softmax(logit, dim = 1, temperature=1))
# %%
