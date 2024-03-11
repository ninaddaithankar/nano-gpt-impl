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
# open and read the dataset
# -------------------------------------------------------------------------------------------------------------------------
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()



# -------------------------------------------------------------------------------------------------------------------------
# tokenize the dataset
# -------------------------------------------------------------------------------------------------------------------------
tokens = sorted(set(text))
vocab_size = len(tokens)

stoi = {s:i for i, s in enumerate(tokens)}
itos = {i:s for i, s in enumerate(tokens)}

encode = lambda s: [stoi[ch] for ch in s]
decode = lambda i: ''.join([itos[idx] for idx in i])



# -------------------------------------------------------------------------------------------------------------------------
# create train and validation splits
# -------------------------------------------------------------------------------------------------------------------------
data = torch.tensor(encode(text), dtype = torch.long)
split_point = int(0.9*len(data))
train_data = data[:split_point]
val_data = data[split_point:]



# -------------------------------------------------------------------------------------------------------------------------
# helper function to get a batch from dataset
# -------------------------------------------------------------------------------------------------------------------------
def get_batch(split):
    data = train_data if split == 'train' else val_data
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i+block_size] for i in ix])
    y = torch.stack([data[i+1 : i+block_size + 1] for i in ix])

    return x.to(device), y.to(device)



# -------------------------------------------------------------------------------------------------------------------------
# helper function to estimate loss in training loop
# -------------------------------------------------------------------------------------------------------------------------
@torch.no_grad
def estimate_loss():
    avg_losses = {}
    
    model.eval()

    for split_mode in ['train', 'val']:
        losses = torch.zeros(eval_iterations)

        for i in range(eval_iterations):
            X, Y = get_batch(split_mode)
            logits, loss = model(X, Y)
            losses[i] = loss

        avg_losses[split_mode] = losses.mean()
    
    model.train()

    return avg_losses



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
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # make a forward pass
            logits, loss = self(idx)

            # take out the logits for last time step
            logits = logits[:, -1, :]    # B, T, C  ->  B, C

            # get the probabilities from the logits
            probs = F.softmax(logits, dim=-1)

            # sample the next char from probs distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # add it to the T+1 timestep in input indexes
            idx = torch.cat((idx, idx_next), dim = 1)

        return idx


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

    # if eval interval, run loss estimation
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, validation loss {losses['val']:.4f} ")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()



# -------------------------------------------------------------------------------------------------------------------------
# basic driver code
# -------------------------------------------------------------------------------------------------------------------------
# batches = torch.stack([train_data[i: i+block_size] for i in torch.randint(0, vocab_size,(4,))])
# print([decode(batch) for batch in batches.tolist()])
# BigramLanguageModel(vocab_size).forward(batches, targets=batches)

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))










