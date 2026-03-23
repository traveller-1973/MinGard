import math
import sys
from engine.Tensor import Tensor
import optim.optimizer as optim
import nn.module as nn
from utils import get_device
from gpt import gpt
import random

class TrainDataset:
    def __init__(self, data, context_length):
        self.data = data
        self.context_length = context_length

    def __len__(self):
        return len(self.data) - self.context_length - 1

    def __getitem__(self, i):
        input_data = self.data[i:i+self.context_length]
        target_data = self.data[i+1:i+self.context_length+1]
        return (input_data, target_data)

def create_dataloader(dataset, batch_size, shuffle=True):
    # 生成所有样本的索引列表
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)

    for batch_start in range(0, len(indices), batch_size):
        # 生成一个batch的索引
        batch_indices = indices[batch_start:batch_start+batch_size]
        batch_data = [dataset[i] for i in batch_indices]
        input_data, target_data = zip(*batch_data)
        yield Tensor(list(input_data)), Tensor(list(target_data))


class ProgressBar:
    def __init__(self, total, prefix="", length=30):
        self.total = max(total, 1)
        self.prefix = prefix
        self.length = length

    def update(self, step, avg_loss):
        ratio = min(max(step / self.total, 0), 1)
        filled = int(self.length * ratio)
        bar = "█" * filled + "-" * (self.length - filled)
        message = (
            f"\r{self.prefix} |{bar}| {step}/{self.total} "
            f"avg_loss: {avg_loss:.4f}"
        )
        sys.stdout.write(message)
        sys.stdout.flush()
        if step >= self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def close(self):
        sys.stdout.write("\n")
        sys.stdout.flush()


with open('./data/天龙八部.txt', 'r',encoding='utf-8') as f:
    text = f.read()
# print(text[:1000])

chars=sorted(list(set(text)))

char_to_idx={ch:i for i,ch in enumerate(chars)}
idx_to_char={i:ch for i,ch in enumerate(chars)}
def encode(text):
    return [char_to_idx[ch] for ch in text]
def decode(text):
    return ''.join([idx_to_char[i] for i in text])

data=Tensor(encode(text),dtype='long')
n=int(len(data)*0.9)
train_data=data[:n]
val_data=data[n:]

batch_size=8
context_length=512
d_model=512
n_head=8
vocab_size=len(chars)
device="cuda:0"
lr=5e-4
d,_=get_device(device)
temperature=0.8
top_k=40
preview_tokens=120
start_prompt="开始"

train_set = TrainDataset(train_data, context_length)
num_batches = math.ceil(len(train_set) / batch_size)


model=gpt(vocab_size,context_length,d_model,n_head).to(device)
optimizer=optim.SGD(model.parameters(),lr,device=device)

num_epochs=10
for epoch in range(num_epochs):
    # model.train()
    total_loss=0
    progress = ProgressBar(
        total=num_batches,
        prefix=f"Epoch [{epoch+1}/{num_epochs}]"
    )
    train_dataloader = create_dataloader(
        train_set, batch_size=batch_size, shuffle=True
    )
    for batch_idx,(input_data,target_data) in enumerate(train_dataloader, start=1):
        input_data=input_data.to(device)
        target_data=target_data.to(device)

        logits,loss=model(input_data,target_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step_loss = float(loss.data)
        total_loss += step_loss
        avg_loss = total_loss / batch_idx
        progress.update(batch_idx, avg_loss)
        # 每25%保存一次模型 
        if batch_idx % (num_batches // 4) == 0:
            nn.save_state_dict(model, f"checkpoints/gpt_epoch{epoch+1}_step{batch_idx}.pkl")

    nn.save_state_dict(model, f"checkpoints/gpt_epoch{epoch+1}.pkl")

    start_tokens = Tensor([encode(start_prompt)], dtype='long').to(device)
    eos_token_id = char_to_idx.get('\n')
    generated = model.generate_(
        start_tokens,
        max_tokens=preview_tokens,
        greedy=False,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=eos_token_id,
    )
    preview_text = decode(generated.data[0].tolist())
    print(f"\n[Epoch {epoch+1}] top-k preview:\n{preview_text}\n")