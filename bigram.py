import torch
import torch.nn as nn

from torch.nn import functional as F

# -------------------------------------------------------------------------------------------------------------------------
# define hyperparams 
# -------------------------------------------------------------------------------------------------------------------------
batch_size = 32
block_size = 8

max_iters = 3000
eval_interval = 200
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iterations = 200



# -------------------------------------------------------------------------------------------------------------------------
# set manual seed for reproducibility
# -------------------------------------------------------------------------------------------------------------------------
torch.manual_seed(1337)



# -------------------------------------------------------------------------------------------------------------------------
# open the dataset
# -------------------------------------------------------------------------------------------------------------------------
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()



# -------------------------------------------------------------------------------------------------------------------------
# process the dataset
# -------------------------------------------------------------------------------------------------------------------------
tokens = sorted(set(text))
vocab_size = len(tokens)

stoi = {s:i for i, s in enumerate(tokens)}
itos = {i:s for i, s in enumerate(tokens)}

encode = lambda s: [stoi[ch] for ch in s]
decode = lambda i: ''.join([itos[idx] for idx in i])



# -------------------------------------------------------------------------------------------------------------------------
# divide the dataset
# -------------------------------------------------------------------------------------------------------------------------
data = torch.tensor(encode(text), dtype = torch.long)
split_point = int(0.9*len(data))
train_data = data[:split_point]
val_data = data[split_point:]



# -------------------------------------------------------------------------------------------------------------------------
# get batch from dataset
# -------------------------------------------------------------------------------------------------------------------------
def get_batch(split):
    data = train_data if split == 'train' else val_data
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i+block_size] for i in ix])
    y = torch.stack([data[i+1 : i+block_size + 1] for i in ix])

    return x.to(device), y.to(device)


# -------------------------------------------------------------------------------------------------------------------------
# define a super duper simple bigram language model for dummies like me
# -------------------------------------------------------------------------------------------------------------------------
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets = None):
        logits = self.token_embedding_table(idx)
        
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    


# -------------------------------------------------------------------------------------------------------------------------
# init the model
# -------------------------------------------------------------------------------------------------------------------------
model = BigramLanguageModel(vocab_size).to(device)



# -------------------------------------------------------------------------------------------------------------------------
# create a PyTorch optimizer
# -------------------------------------------------------------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)



# -------------------------------------------------------------------------------------------------------------------------
# train the model
# -------------------------------------------------------------------------------------------------------------------------
for iter in range(max_iters):
    xb, yb = get_batch('train')

    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    optimizer.step()



# -------------------------------------------------------------------------------------------------------------------------
# basic driver code
# -------------------------------------------------------------------------------------------------------------------------
batches = torch.stack([train_data[i: i+block_size] for i in torch.randint(0, vocab_size,(4,))])
print([decode(batch) for batch in batches.tolist()])
BigramLanguageModel(vocab_size).forward(batches, targets=batches)










