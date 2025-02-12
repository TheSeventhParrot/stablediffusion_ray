import torch

# Clear the cache
torch.cuda.empty_cache()

# To check memory usage before and after
print(torch.cuda.memory_allocated())  # Bytes allocated
print(torch.cuda.memory_reserved())   # Bytes reserved
