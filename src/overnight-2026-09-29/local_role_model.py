"""Amount/date role head with explicit local-context pooling; zero residual init."""
import sys,math
from pathlib import Path
P=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(P/'natural-v4'))
from joint_model import Joint as Original,CLASSES,candidates,predict,t
import torch
from torch import nn
from torch.nn import functional as F
class Joint(Original):
 def __init__(self):
  super().__init__();self.local_role_head=nn.Linear(self.encoder.config.hidden_size,len(CLASSES));nn.init.zeros_(self.local_role_head.weight);nn.init.zeros_(self.local_role_head.bias)
 def forward(self,x):
  h=self.encoder(**x).last_hidden_state;mask=x['attention_mask'].bool();att=torch.einsum('bld,fd->bfl',h,self.queries)/math.sqrt(h.shape[-1]);att=att.masked_fill(~mask[:,None,:],-1e4);pooled=torch.einsum('bfl,bld->bfd',att.softmax(-1),h);cls=[head(pooled[:,i]) for i,head in enumerate(self.heads)]
  # Masked 33-token neighborhood, independent of other documents' padding lengths.
  z=(h*mask[:,:,None]).transpose(1,2);total=F.avg_pool1d(z,33,stride=1,padding=16);count=F.avg_pool1d(mask[:,None,:].to(h.dtype),33,stride=1,padding=16).clamp_min(1/33);local=(total/count).transpose(1,2)
  return cls,self.role_head(h)+self.local_role_head(local)
