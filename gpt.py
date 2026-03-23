from engine.Tensor import Tensor
import nn.module as nn
import random
from warnings import warn
import nn.act_func as F
from nn.loss_func import cross_entropy_loss

    
class embedding(nn.Module):
    def __init__(self, vocab_size,context_length,d_model):
        super(embedding, self).__init__()
        self.embedding=nn.Embedding(vocab_size,d_model)
        self.positonal_embedding=nn.Embedding(context_length,d_model)
        # self.lm_head=nn.Linear(d_model,vocab_size)

    def __call__(self,src):
        # src,trg: (batch_size, context_length)

        batch_size,context_length=src.shape

        word_embedidng=self.embedding(src) #(batch_size, context_length, d_model)
        pos_embedding=self.positonal_embedding(src.d.arange(context_length)) #(context_length, d_model)
        embedding=word_embedidng+pos_embedding #(batch_size, context_length, d_model)

        return embedding

class layer_norm(nn.Module):
    def __init__(self, d_model):
        super(layer_norm, self).__init__()
        self.gamma=Tensor(self.d.ones(d_model), device=self.device, requires_grad=True)
        self.beta=Tensor(self.d.zeros(d_model), device=self.device, requires_grad=True)

    def __call__(self, x):
        # x: (batch_size, context_length, d_model)
        mean=x.mean(-1,keepdims=True)
        var=x.var(-1,keepdims=True)
        out=self.gamma*((x-mean)/(var+1e-12)**0.5)+self.beta
        # out: (batch_size, context_length, d_model)

        return out

class head(nn.Module):
    def __init__(self, d_model, d_head, context_length):
        super(head, self).__init__()
        self.q=nn.Linear(d_model,d_head)
        self.k=nn.Linear(d_model,d_head)
        self.v=nn.Linear(d_model,d_head)
        # self.out=nn.Linear(d_model,d_model)

        # 下三角矩阵
        self.mask=self.d.tril(self.d.ones((context_length,context_length)))

    def __call__(self, x):
        # x: (batch_size, context_length, d_model)
        
        batch_size,context_length,d_model=x.shape

        # q,k,v: (batch_size, context_length, d_head)
        q=self.q(x)
        k=self.k(x)
        v=self.v(x)

        d_head = q.shape[-1]
        score=q @ k.transpose(-2,-1) *d_head**-0.5
        score=score.masked_fill(self.mask[:context_length,:context_length]==0,float('-inf'))
        score=F.softmax(score,dim=-1)
        # score: (batch_size, context_length, context_length)

        out=score @ v
        # out: (batch_size, context_length, d_head)

        return out
    
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head, context_length):
        super(MultiHeadAttention, self).__init__()

        # 创建n_head个head，其中d_head=d_model//n_head
        self.heads=nn.ModuleList([head(d_model,d_model//n_head,context_length) for _ in range(n_head)])
        self.out=nn.Linear(d_model,d_model)

    def __call__(self, x):
        # x: (batch_size, context_length, d_model)
        batch_size,context_length,d_model=x.shape

        out=Tensor.cat([head(x) for head in self.heads],dim=-1)
        # out: (batch_size, context_length, d_model)
        out=self.out(out)
        # out: (batch_size, context_length, d_model)

        return out
    
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super(FeedForward, self).__init__()
        self.fc1=nn.Linear(d_model,d_model*4)
        self.fc2=nn.Linear(d_model*4,d_model)

    def __call__(self, x):
        # x: (batch_size, context_length, d_model)
        out=F.relu(self.fc1(x))
        out=self.fc2(out)
        # out: (batch_size, context_length, d_model)

        return out
    
class block(nn.Module):
    def __init__(self, d_model, n_head, context_length):
        super(block, self).__init__()
        self.self_attention=MultiHeadAttention(d_model,n_head,context_length)
        self.feed_forward=FeedForward(d_model)
        self.ln1=layer_norm(d_model)
        self.ln2=layer_norm(d_model)

    def __call__(self, x):
        # x: (batch_size, context_length, d_model)
        # 加入残差连接
        out=x+self.self_attention(self.ln1(x))
        out=out+self.feed_forward(self.ln2(out))
        # out: (batch_size, context_length, d_model)

        return out

class gpt(nn.Module):
    def __init__(self, vocab_size,context_length,d_model,n_head):
        super(gpt,self).__init__()

        self.context_length=context_length
        self.embedding=embedding(vocab_size,context_length,d_model)
        self.blocks=nn.ModuleList([block(d_model,n_head,context_length) for _ in range(8)])
        self.lm_head=nn.Linear(d_model,vocab_size)

    def __call__(self,src,trg=None):
        # src,trg: (batch_size, context_length)
        embedding=self.embedding(src)
        # embedding: (batch_size, context_length, d_model)

        out=embedding
        for block in self.blocks:
            out=block(out)
        # out: (batch_size, context_length, d_model)
            
        logits=self.lm_head(out)
        # logits: (batch_size, context_length, vocab_size)
        
        if trg is None:
            loss=None
        else:
            logits=logits.view(logits.shape[0]*logits.shape[1],-1)
            target=trg.view(-1)
            loss=cross_entropy_loss(logits,target)
        return logits,loss

    def generate_(
        self,
        idx: Tensor,
        max_tokens: int,
        temperature: float = 1.0,
        greedy: bool = False,
        top_k: int = None,
        eos_token_id: int = None,
    ) -> Tensor:
        """自回归地产生后续token。

        参数:
            idx: 初始上下文，形状为(batch_size, seq_len)。
            max_tokens: 需要额外生成的token数量。
            temperature: 采样温度，>0，数值越大越随机。
            greedy: 是否使用贪心策略（选择概率最大token），默认False表示按概率采样。
            top_k: 仅保留概率最高的前k个候选token进行采样，None表示不启用。
            eos_token_id: 结束token id，若提供则当batch内样本均生成该token后提前停止。

        返回:
            包含生成结果的新Tensor，形状为(batch_size, seq_len + max_tokens)。
        """
        if max_tokens <= 0:
            return idx
        if temperature <= 0:
            raise ValueError("temperature must be greater than 0.")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be greater than 0 when provided.")

        if idx.device != self.device:
            idx = idx.to(self.device)

        grad_state = Tensor.grad_enabled
        Tensor.grad_enabled = False
        try:
            finished = None
            for _ in range(max_tokens):
                if idx.shape[1] > self.context_length:
                    idx_cond = idx[:, -self.context_length:]
                else:
                    idx_cond = idx

                logits, _ = self(idx_cond)
                logits = logits[:, -1, :]
                vocab_size = logits.shape[-1]

                if temperature != 1.0:
                    logits = logits / temperature

                if top_k is not None and top_k < vocab_size:
                    keep_k = min(top_k, vocab_size)
                    logits_data = logits.data
                    kth = logits.d.partition(logits_data, -keep_k, axis=-1)[:, -keep_k][:, None]
                    filtered_logits = logits_data.copy()
                    filtered_logits[logits_data < kth] = -float("inf")
                    logits = Tensor(filtered_logits, device=logits.device, dtype=logits.dtype)

                probs = F.softmax(logits, dim=-1)
                batch_size = probs.shape[0]
                if eos_token_id is not None and finished is None:
                    finished = [False] * batch_size

                next_indices = []
                for b in range(batch_size):
                    if finished is not None and finished[b]:
                        next_idx = int(eos_token_id)
                        next_indices.append(next_idx)
                        continue

                    prob = probs.data[b]
                    prob = prob / (prob.sum() + 1e-12)

                    if greedy:
                        next_idx = int(prob.argmax())
                    else:
                        next_idx = int(probs.d.random.choice(vocab_size, p=prob))
                    next_indices.append(next_idx)

                    if finished is not None and next_idx == int(eos_token_id):
                        finished[b] = True

                next_tokens_array = probs.d.asarray(next_indices, dtype=idx.data.dtype).reshape(batch_size, 1)
                next_tokens = Tensor(next_tokens_array, device=idx.device, dtype=idx.data.dtype)
                idx = Tensor.cat([idx, next_tokens], dim=1)

                if finished is not None and all(finished):
                    break
        finally:
            Tensor.grad_enabled = grad_state

        return idx
    