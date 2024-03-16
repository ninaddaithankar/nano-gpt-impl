import os
import torch
import torch.nn as nn

from torch.nn import functional as F

# -------------------------------------------------------------------------------------------------------------------------
# define hyperparams 
# -------------------------------------------------------------------------------------------------------------------------
batch_size = 16
block_size = 64
embedding_size = 64

max_iters = 2000
eval_interval = 500
checkpoint_interval = 1000
learning_rate = 5e-4
eval_iterations = 100
dropout_percent = 0.1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

CHECKPOINT_PATH = './checkpoints/latest_checkpoint.pth'



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
# define a single head attention module
# -------------------------------------------------------------------------------------------------------------------------
class Head(nn.Module):
    """single head of self attention"""

    def __init__(self, head_size):
        super().__init__()

        self.key   = nn.Linear(embedding_size, head_size, bias=False)
        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout_percent)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v

        return out
    



# -------------------------------------------------------------------------------------------------------------------------
# create a multi-head attention module that will run multiple smaller single head attention modules in parallel
# -------------------------------------------------------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size) -> None:
        super().__init__()

        self.heads = nn.ModuleList(Head(head_size) for _ in range(num_heads))
        self.projection = nn.Linear(embedding_size, embedding_size)

        self.dropout = nn.Dropout(dropout_percent)


    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.projection(out)
        out = self.dropout(out)
        
        return out
    
    


# -------------------------------------------------------------------------------------------------------------------------
# create a feed-forward layer for letting the model think
# -------------------------------------------------------------------------------------------------------------------------
class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout_percent)
        )

    def forward(self, x):
        return self.net(x)
    



# -------------------------------------------------------------------------------------------------------------------------
# create a decoder block as seen in the transformer paper
# -------------------------------------------------------------------------------------------------------------------------
class Block(nn.Module):

    def __init__(self, n_embd, n_heads):
        super().__init__()

        meta_nembd = n_embd//n_heads

        self.self_attention_heads = MultiHeadAttention(n_heads, meta_nembd)
        self.feed_forward = FeedForward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.self_attention_heads(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))

        return x
    



# -------------------------------------------------------------------------------------------------------------------------
# define a super duper simple bigram language model for dummies like me
# -------------------------------------------------------------------------------------------------------------------------
class BigramLanguageModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding_table = nn.Embedding(vocab_size, embedding_size)

        self.decoder_blocks = nn.Sequential(
            Block(embedding_size, 4),
            Block(embedding_size, 4),
            Block(embedding_size, 4),
            nn.LayerNorm(embedding_size)
        )
        self.language_model_head = nn.Linear(embedding_size, vocab_size)


    def forward(self, idx, targets = None):
        token_embeddings = self.token_embedding_table(idx)
        postion_embeddings = self.position_embedding_table(idx)

        x = token_embeddings + postion_embeddings
        x = self.decoder_blocks(x)
        logits = self.language_model_head(x)
        
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
model = BigramLanguageModel().to(device)



# -------------------------------------------------------------------------------------------------------------------------
# create a PyTorch optimizer
# -------------------------------------------------------------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


def checkpoint(model_state_dict, optim_state_dict, epoch, losses):
    state = {
        'epoch' : epoch,
        'model_state_dict': model_state_dict,
        'optim_state_dict': optim_state_dict,
        'losses': losses
    }

    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')

    torch.save(state, CHECKPOINT_PATH)


# -------------------------------------------------------------------------------------------------------------------------
# define the training loop
# -------------------------------------------------------------------------------------------------------------------------
def train(iterations, prev_iterations = 0):
    model.train()

    losses = {}
    for iter in range(prev_iterations, prev_iterations + iterations + 1):

        # if eval interval, run loss estimation
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, validation loss {losses['val']:.4f} ")
            
        # if checkpoint interval, save model/optimizer weights and current losses
        if iter % checkpoint_interval == 0:
            checkpoint(model.state_dict(), optimizer.state_dict(), iter, losses)
            print(f"======= checkpoint saved at iter: {iter} =======")

        xb, yb = get_batch('train')

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()




# -------------------------------------------------------------------------------------------------------------------------
# define the inference loop
# -------------------------------------------------------------------------------------------------------------------------
def blabber(idx, max_new_tokens):
    model.eval()

    for _ in range(max_new_tokens):
        # crop tokens to block size
        idx_block = idx[:, -block_size:]

        # make a forward pass
        logits, loss = model(idx_block)

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
# ready, set, goo...
# -------------------------------------------------------------------------------------------------------------------------

if os.path.exists(CHECKPOINT_PATH):
    state = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optim_state_dict'])
    epoch = state['epoch']
    losses = state['losses']

    print(f"Loaded from checkpoint on iter: {epoch} with train loss {losses['train']:.4f}, validation loss {losses['val']:.4f}")
    print("Resuming model training...")
    train(max_iters, epoch)

else:
    train(max_iters)


context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(blabber(context, max_new_tokens=1000)[0].tolist()))










