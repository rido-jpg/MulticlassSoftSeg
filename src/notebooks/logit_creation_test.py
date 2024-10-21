import torch

logits = torch.rand(4,2,2)

print(logits.shape)
print(logits)

logits_v11 = logits[[3]].clone().sum(dim=0, keepdim=True)
logits_v12 = logits[[0,1,2]].clone().sum(dim=0, keepdim=True)

print(logits_v11.shape)
print(logits_v11)
print(logits_v12.shape)
print(logits_v12)

logits_v21 = logits[[3]].sum(dim=0, keepdim=True)
logits_v22 = logits[[0,1,2]].sum(dim=0, keepdim=True)

print(logits_v21.shape)
print(logits_v21)
print(logits_v22.shape)
print(logits_v22)

logits_v31 = logits[[3]].sum(dim=0, keepdim=True).clone()
logits_v32 = logits[[0,1,2]].sum(dim=0, keepdim=True).clone()

print(logits_v31.shape)
print(logits_v31)
print(logits_v32.shape)
print(logits_v32)

logits_v41 = logits.clone()[[3]].sum(dim=0, keepdim=True)
logits_v42 = logits.clone()[[0,1,2]].sum(dim=0, keepdim=True)

print(logits_v41.shape)
print(logits_v41)
print(logits_v42.shape)
print(logits_v42)

print(logits)
print(torch.eq(logits_v11,logits_v21))
print(torch.eq(logits_v11,logits_v31))
print(torch.eq(logits_v31,logits_v21))
print(torch.eq(logits_v11, logits_v41))

print(torch.eq(logits_v12,logits_v22))
print(torch.eq(logits_v12,logits_v32))
print(torch.eq(logits_v32,logits_v22))
print(torch.eq(logits_v12, logits_v42))