import torch
a = torch.tensor([[1,5,62,54], [2,6,2,6], [2,65,2,6]])
r = torch.rand(5,1,10)
print(r)
r = r.log_softmax(dim=-1)

print(r)
print(a.shape)
_,r = r.max(2) # .max(2) 返回2个结果，每行最大数值，以及最大数值对应的数组索引
print(_)

print(r)