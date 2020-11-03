import torch
from torch import nn

embedding = nn.Embedding(1, 4) # 假定字典中只有5个词，词向量维度为4
word = [[0.1, -0.1, 0.1,0.1,0.1,0.1],
        [0.1, 0.1, 0.1,0.1,0.1,0.1]] # 每个数字代表一个词，例如 {'!':0,'how':1, 'are':2, 'you':3,  'ok':4}
                    #而且这些数字的范围只能在0～4之间，因为上面定义了只有5个词
embed = embedding(torch.LongTensor(word))
print(embed) 
print(embed.size())