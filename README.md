# MinGard

基于numpy和cupy实现的类似pytorch的自动微分训练框架

- 仅用numpy和cupy实现的自动微分引擎
- 支持Tensor运算
- 自定义神经网络网络模型构建
- 基于MinGard的字符级gpt实现
- ......



### MinGard Demo

~~~python
from engine.Tensor import Tensor
import nn.module as nn
import nn.act_func as F
from optim.optimizer import SGD

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel,self).__init__()
        self.embed=nn.Embedding(20,10) 
        self.ln=nn.Linear(10,20)
        self.ln2=nn.Linear(20,1)
    def __call__(self,x):
        # x: [batch_size,seq_len]

        x=self.embed(x)
        x=self.ln(x) 
        x=F.relu(x)
        x=self.ln2(x)
        return x

model=TestModel()
optimizer=SGD(model.parameters(),lr=0.001)

x=Tensor([[1,2,3,4,5],[6,7,8,9,10]])
y=model(x)   # y.shape:(2, 5, 1)
y.backward()
optimizer.step()
~~~

### Requirements

~~~
numpy
cupy
~~~

