import torch
from torch_scatter import segment_csr
import random

# mask= torch.tensor([0,0,0,0],  dtype=torch.int32)

# print(torch.where(mask>0)[0][0])

tensor = torch.tensor([-1])

if tensor==-1:
    print(1)
else:
    print(0)


print(torch.randint(0, 5, (20,), dtype=torch.int32))
print(random.randint(1,3))
print(random.randint(1,3))
print(random.randint(1,3))


pi = torch.tensor([-1,-1,0,2,0,4,1,1,4,9,26,-1,0])

_, pi2 = torch.unique(pi, return_inverse=True)

print(pi2)

ptr= torch.tensor([0,3,6,6,10])
src = torch.tensor([2,3,4,2,2,5,4,2,2,9])

print(torch.sort(src))
a, ind = torch.sort(src)
_,ind = torch.sort(ind)
print(a[ind])

print(segment_csr(src,ptr, reduce="max"))

src = torch.tensor([-1,-1,19,12,3,3,2,4,1,1])
print(torch.unique_consecutive(src, return_counts=True))
print(torch.unique(src, return_inverse=True))
