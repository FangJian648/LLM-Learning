import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self,dim,head):
        super(MultiHeadAttention,self).__init__()
        self.head=head
        self.d_k=dim//head
        self.w_q=nn.Linear(dim,self.d_k*head)
        self.w_k=nn.Linear(dim,self.d_k*head)
        self.w_v=nn.Linear(dim,self.d_k*head)
        self.w_o=nn.Linear(self.d_k*head,dim)
        self.softmax=nn.Softmax(dim=-1)

    def forward(self,x,mask=None):
        # x (batch,seq,dim)
        batch,seq,dim=x.size()
        q=self.w_q(x)
        k=self.w_k(x)
        v=self.w_v(x)
        q=q.view(batch,seq,self.head,-1).transpose(1,2)
        k=k.view(batch,seq,self.head,-1).transpose(1,2)
        v=v.view(batch,seq,self.head,-1).transpose(1,2)

        score=torch.matmul(q,k.transpose(-1,-2))/math.sqrt(dim)

        if mask is not None:
            mask=mask.unsqueeze(1)
            score.masked_fill(mask,-1e9)
        score = self.softmax(score)
        
        atten=torch.matmul(score,v)
        atten=atten.transpose(1,2).contiguous().view(batch,seq,-1)
        atten=self.w_o(atten)

        return atten

dim =64
head=8
batch=2
seq=5

x=torch.randn(batch,seq,dim)
mha=MultiHeadAttention(dim,head)
atten=mha(x)
print(atten)