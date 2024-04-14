import math

class Value:
    def __init__(self,data,_children=(),_op='',label=''):
        self.data = data
        self.grad = 0.0
        # 定义反向传播函数，叶子节点（直接用Value定义的节点）的_backward是None,中间节点在计算时定义相关的_backward
        # _backward用于计算所有子节点的梯度
        self._backward = lambda: None

        # 这个值来自哪里 
        self._prev = set(_children)
        # 这个值是通过哪个操作得到的
        self._op = _op
        # 这个值的标签
        self.label = label

    def __repr__(self):
        return f'Value(data={self.data})'
    
    # 加法
    def __add__(self,other):
        # 处理Value+a的情况
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data+other.data,(self,other),'+')
        def _backward():
            # 链式求导的局部梯度
            # 链式求导梯度逐个相乘，根节点即最终变量的导数为1：o=f(x),o为最终变量，do/dx=(f(x+h)-f(x))/h,do/do=1
            # out=a+b,a是self，b是other，a的导数是1*out.grad，b的导数是1*out.grad
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
        out._backward = _backward
        return out
    
    def __radd__(self,other):
        # 交换律，处理a+Value的情况
        return self.__add__(other)
    
    # 负号
    def __neg__(self):
        return self * -1
    
    # 减法
    def __sub__(self,other):
        return self + (-other)
    
    # 乘法
    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data*other.data,(self,other),'*')
        def _backward():
            # out=a*b,a的梯度是b*out.grad，b的梯度是a*out.grad
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self,other):
        return self.__mul__(other)
    
    # 除法
    def __truediv__(self,other):
        return self*other**-1
    
    # 幂
    def __pow__(self,other):
        assert isinstance(other,(int,float))
        out = Value(self.data**other,(self,),f'**{other}')
        def _backward():
            # out=a**b,a的梯度是(b*a**(b-1))*out.grad
            self.grad += other * self.data**(other-1) * out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        e=math.exp(self.data)
        out = Value(e,(self,), 'exp')
        def _backward():
            # out=exp(x),x的梯度是exp(x)*out.grad
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x=self.data
        t=(math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t,(self,), 'tanh')
        def _backward():
            # out=tanh(x),x的梯度是(1-tanh(x)**2)*out.grad
            self.grad += (1-t**2) * out.grad
        out._backward = _backward
        return out

    def backword(self):
        topo=[]
        # 对变量传播图进行拓扑排序，确保反向传播时该变量后的变量的导数已经计算完毕
        visited=set()
        def build_topo(n):
            if n not in visited:
                visited.add(n)
                for child in n._prev:
                    build_topo(child)
                # 所有子节点处理完后，将自己加入到topo中，即topo中任意节点都在其子节点的后面
                topo.append(n)

        build_topo(self)  #以自己为根节点进行拓扑排序
        self.grad = 1.0    #根节点导数为1
        for n in reversed(topo): #从根节点开始反向传播
            n._backward()
