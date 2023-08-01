### Transformer for Time Series V1 ###
### Input: [N, L, 1] values of function at time t
### Output: [N, L, 1]


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import copy
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Positional Embedding ###

def positional_encoding(length, depth):
  depth = depth/2

  positions = torch.arange(length).unsqueeze(1)    # (seq, 1)
  depths = torch.arange(depth).unsqueeze(0)/depth   # (1, depth)

  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = torch.cat([torch.sin(angle_rads), torch.cos(angle_rads)],axis=-1) 
  pos_encoding = pos_encoding.to(device)

  return pos_encoding # [L, d_model]


class Embedder(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.embed_linear = nn.Linear(input_dim, embed_dim)
    
    def forward(self, x):
        # x from [N,L, input_dim], [N,S, input_dim] to [N,L,E], [N,S,E]
        return self.embed_linear(x) # embedded x of the form [N,L,E], [N,S,E]


class Positional_Embedding(nn.Module):
    def __init__(self, input_dim, max_len_samples, d_model):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        self.embedding = Embedder(input_dim=self.input_dim, embed_dim=self.d_model)
        self.pos_encoding = positional_encoding(length=max_len_samples, depth=d_model)

    def forward(self, x):

        len_seq_x = x.shape[-2]
        x = self.embedding(x) # [N, L, input_dim] --> [N, L, d_model] 
        x = x*self.d_model**(0.5) # div by sqrt(d_k)
        x = x + self.pos_encoding[:len_seq_x, :].unsqueeze(-3)
        return x



### Attention Classes ###

class Base_Attention(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.mha = nn.MultiheadAttention(**kwargs)
        self.layer_norm = nn.LayerNorm(normalized_shape=kwargs['embed_dim'])
    
    def look_ahead_mask(self, dim1:int, dim2:int) -> torch.FloatTensor:
        mask = torch.triu(torch.ones(dim1,dim2), diagonal=1)
        mask = mask.to(device)
        mask[mask.bool()] = -float('inf')
        return mask

    def sequence_mask(seq:torch.LongTensor, padding_idx:int=None) -> torch.BoolTensor:
        """ seq: [N, seq_len], which is padded with an index called padding_idx.     
        if seq == padding_index, returns True, and corresponding value is being ignored during attention """
        return seq == padding_idx


class Cross_Attention(Base_Attention):

    def forward(self, x, context):

        causal_mask = self.look_ahead_mask(dim1=x.shape[-2], dim2=context.shape[-2])

        attn_output, attn_scores = self.mha(
        query=x,
        key=context,
        value=context,
        attn_mask = causal_mask,
        need_weights=True)

        self.last_attn_scores = attn_scores

        x = x + attn_output
        x = self.layer_norm(x)

        return x


class Self_Attention(Base_Attention):

    def forward(self, x):
        attn_output, _ = self.mha(
        query=x,
        key=x,
        value=x)

        x = x + attn_output
        x = self.layer_norm(x)

        return x


class Causal_Self_Attention(Base_Attention):

    def forward(self, x):

        causal_mask = self.look_ahead_mask(dim1=x.shape[-2], dim2=x.shape[-2])

        attn_output, _ = self.mha(
        query=x,
        key=x,
        value=x,
        attn_mask = causal_mask
        )

        x = x + attn_output
        x = self.layer_norm(x)

        return x




### FEB ###

class Kernel_Layer(nn.Module):
    """ Custom Kernel layer for production operator """
    def __init__(self, m, d_in, d_out):
        super().__init__()
        self.d_in, self.d_out, self.m = d_in, d_out, m
        self.weights = torch.Tensor(self.m, self.d_in, self.d_out) # [M, Din, Dout]
        self.weights = nn.Parameter(self.weights)  # nn.Parameter is a Tensor that's a module parameter.

        # initialize weights and biases
        nn.init.normal_(self.weights) # weight init

    def forward(self, x): # x of the form [N,M,Din]
        x = x.unsqueeze(-1).repeat(1,1,1,self.d_out) # [N,M,Din,repeat Dout times]
        y = x*self.weights.unsqueeze(0) #[N,M,Din,Dout]
        y = torch.sum(y, dim=-2) # [N, M, Dout]
        return y  


class FEB_f(nn.Module):
    def __init__(self, m, d_model):
        super().__init__()
        self.m = m
        self.d_model = d_model
        self.linear_feb = Kernel_Layer(m=2*m, d_in=d_model, d_out=d_model)
    
    def forward(self, x): # x of the form [N,L,D]
        L = x.shape[-2]
        x = torch.fft.fft(x, n=self.m, dim=-2) # [N,M,D]

        x = torch.cat((torch.real(x), torch.imag(x)), dim = -2) # [N, 2*M, D]

        x = self.linear_feb(x) # [N,2*M,D]

        x = torch.reshape(x,(x.shape[-3],x.shape[-2]//2, x.shape[-1], 2)) # [N,M,D,2]
        x = torch.view_as_complex(x) # [N,M,D] complex
        
        x = torch.real(torch.fft.ifft(x, n=L, dim=-2)) # [N,L,D] real
        return x



### FFNetwork ###

class FFN(nn.Module):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
        self.seq = nn.Sequential(nn.Linear(d_model, dff), nn.ReLU(), nn.Linear(dff, d_model), nn.Dropout(p=dropout_rate))

    def forward(self, x):
        x = x + self.seq(x) 
        x = self.layer_norm(x)
        return x


####################################################################################################################################################################################################################
####################################################################################################################################################################################################################


### Transformer ###

def get_clones(module, num_layers):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_layers)])

class EncoderLayer(nn.Module):
    def __init__(self, m, num_heads, d_model, dff):
        super().__init__()

        self.feb_f = FEB_f(m=m, d_model=d_model)
        self.self_attn = Self_Attention(num_heads=num_heads, embed_dim=d_model, batch_first = True, dropout=0.1)
        self.ffn = FFN(d_model=d_model, dff=dff)
    
    def forward(self, x):
        x = self.feb_f(x=x)
        x = self.self_attn(x=x)
        x = self.ffn(x=x)
        return x


class Encoder(nn.Module):
    def __init__(self, m, input_dim, num_enc_layers, max_len_samples, d_model, num_heads, dff):
        super().__init__()

        self.d_model = d_model
        self.num_enc_layers = num_enc_layers

        self.embedding = Positional_Embedding(input_dim=input_dim, max_len_samples=max_len_samples, d_model=d_model)

        self.enc_layers = get_clones(EncoderLayer(m=m, num_heads=num_heads, d_model=d_model, dff=dff), num_layers=num_enc_layers)

    def forward(self, x):

        x = self.embedding(x)

        for i in range(self.num_enc_layers):
            x = self.enc_layers[i](x)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, m, num_heads, d_model, dff):
        super().__init__()

        self.feb_f = FEB_f(m=m, d_model=d_model)
        self.cross_attn = Cross_Attention(num_heads=num_heads, embed_dim=d_model, batch_first = True, dropout=0.1)
        self.ffn = FFN(d_model=d_model, dff=dff)
    
    def forward(self, x, context):
        x = self.feb_f(x=x)
        x = self.cross_attn(x=x, context=context)

        self.last_attn_scores = self.cross_attn.last_attn_scores

        x = self.ffn(x=x)
        return x


class Decoder(nn.Module):
    def __init__(self, m, input_dim, num_dec_layers, d_model, num_heads, dff):
        super().__init__()

        self.d_model = d_model
        self.num_dec_layers = num_dec_layers

        self.embedding = Embedder(input_dim=input_dim, embed_dim=d_model)

        self.dec_layers = get_clones(DecoderLayer(m=m, num_heads=num_heads, d_model=d_model, dff=dff), num_layers=num_dec_layers)

    def forward(self, x, context):

        x = self.embedding(x)

        for i in range(self.num_dec_layers):
            x = self.dec_layers[i](x, context)
        
        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        return x


class Transformer(nn.Module):
    def __init__(self, m, input_dim, output_dim, num_enc_layers, num_dec_layers, max_len_samples, d_model, num_heads, dff):
        super().__init__()

        self.encoder = Encoder(m=m, input_dim=input_dim, num_enc_layers=num_enc_layers, max_len_samples=max_len_samples, d_model=d_model, num_heads=num_heads, dff=dff)
        self.decoder = Decoder(m=m, input_dim=input_dim, num_dec_layers=num_dec_layers, d_model=d_model, num_heads=num_heads, dff=dff)
        self.final_layer_x = nn.Linear(d_model, output_dim)

    def forward(self, x, context):

        context = self.encoder(x=context)
        x = self.decoder(x=x, context=context)
        x = self.final_layer_x(x)
        return x # output


####################################################################################################################################################################################################################
####################################################################################################################################################################################################################


### Learning Rate ###

class Scheduler_Optim():
    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0
    
    def step(self):
        self._update_learning_rate()
    
    def _get_lr_scale(self):
        return (self.d_model**(-0.5))*min(self.n_steps**(-0.3), self.n_steps*self.n_warmup_steps**(-1.5))
    
    def _update_learning_rate(self):
        self.n_steps += 1
        lr = self.lr_mul*self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr



### Train ###

class Train_Model(nn.Module):
    def __init__(self, model, optimizer, scheduler, criterion, num_epoch, eps, PATH):
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.num_epoch = num_epoch
        self.eps = eps
        self.path = PATH

    def forward(self, x, context, tgt, Loss = 100):
        # for epoch in range(self.num_epoch):
        epoch=0
        while Loss >= self.eps:

            self.optimizer.zero_grad()

            output = self.model(x,context)

            Loss = self.criterion(output, tgt)

            Loss.backward()

            self.optimizer.step()

            self.scheduler.step()

            epoch+=1

            if epoch%1 == 0 : 
                print("it. ", epoch, "loss ", Loss.item())
        
        # save model parameters after training
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': Loss,
            }, self.path)

        return 


####################################################################################################################################################################################################################
####################################################################################################################################################################################################################


### Data Preparation ###


# function to approximate
def func(x):
    return torch.sin(10*x)+torch.cos(3*x)+torch.sin(5*x)+torch.cos(7*x)

# data generation
def prepare_data(func, lower_b, upper_b, nb_points, stride, L, S):  # nb_points=128, shift=2
    kernel_size = S+L                                               # sampled sequence of total length S+L
    N = torch.tensor((nb_points-(kernel_size-1)-1)/stride + 1).item()   # number of batches available in sequence

    while N%1 != 0:
        nb_points = nb_points + 1
        N = torch.tensor((nb_points-(kernel_size-1)-1)/stride + 1).item()
    
    N = int(torch.floor(torch.tensor((nb_points-(kernel_size-1)-1)/stride + 1)).item())  

    # input sequence ususally given (here generated)
    seq_x = torch.linspace(lower_b, upper_b, nb_points)
    step = seq_x[1:]-seq_x[:-1]
    step = step.mean()
    sequence = func(seq_x) # ex [1, ..., 128]
    
    el_0 = sequence[0:kernel_size]
    input_enc = el_0[:S].unsqueeze(0) # ex: [1, ..., 10]
    input_dec = el_0[S-1:S-1+L].unsqueeze(0)
    tgt_dec = el_0[S:S+L].unsqueeze(0)

    for n in range(N-1):
        el_n = sequence[stride*(n+1):stride*(n+1)+kernel_size]
        input_enc = torch.cat((input_enc, el_n[:S].unsqueeze(0)))
        input_dec = torch.cat((input_dec, el_n[S-1:S-1+L].unsqueeze(0)))
        tgt_dec = torch.cat((tgt_dec, el_n[S:S+L].unsqueeze(0)))

    return input_enc.unsqueeze(-1), input_dec.unsqueeze(-1), tgt_dec.unsqueeze(-1), N, sequence, seq_x, step.item()



####################################################################################################################################################################################################################
####################################################################################################################################################################################################################


### Hyperparameters ###

# Data Samples
lower_b = 0.
upper_b = 4*torch.pi
nb_points = 1024
S = 100
L = 50
stride = 4

# Transformer Setup
input_dim = 1
output_dim = 1
num_enc_layers = 4
num_dec_layers = 4
d_model = 256
num_heads = 8
dff = 512
m = 16

# Scheduler Setup
lr_mul = 0.01
n_warmup_steps = 100

# Training Setup
num_epoch = 1000
eps = 1e-4

# Testing Setup
num_pred_steps = 400

# Saved directory
PATH = "path_to_model/V1_sinus_model.pth"



### Instanciate Classes ###

model = Transformer(m=m, input_dim=input_dim, output_dim=output_dim, num_enc_layers=num_enc_layers, num_dec_layers=num_dec_layers, max_len_samples=S+L, d_model=d_model, num_heads=num_heads, dff=dff)
model.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), betas=(0.9,0.98), eps=1e-9)
scheduler = Scheduler_Optim(optimizer=optimizer, lr_mul=lr_mul, d_model=d_model, n_warmup_steps=n_warmup_steps)
criterion = nn.MSELoss()
train_model = Train_Model(model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion, num_epoch=num_epoch, eps=eps, PATH=PATH)



### Data Generation ###

input_enc, input_dec, tgt_dec, N, sequence, seq_x, step = prepare_data(func=func, lower_b=lower_b, upper_b=upper_b, nb_points=nb_points, stride=stride, L=L, S=S)

input_enc = input_enc.to(device)
input_dec = input_dec.to(device)
tgt_dec = tgt_dec.to(device)

# N = N.to(device)
# sequence = sequence.to(device)
# seq_x = seq_x.to(device)
# step = step.to(device)

### Train ###

# train_model(x=input_dec, context=input_enc, tgt=tgt_dec)



### Results ###

# Load Model & Parameters

checkpoint = torch.load(PATH, map_location=torch.device(device))
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']

model.eval()

# Generate loop for predicting future values
    # Input: last couple S+L
    # Output: tgt
    # new input: S+L-1,tgt
    # new output: tgt+1 ...

def prediction_n_step(n, input_dec, input_enc): #n: number of prediction steps
    context = input_enc[-1,:,:].unsqueeze(0)    # [1,S,1]
    x = input_dec[-1,:,:].unsqueeze(0)          # [1,L,1]

    list_pred = []

    for i in range(n):
        pred = model(x=x, context=context)                              # [1,L,1]
        pred = pred[0,-1,:]
        list_pred.append(pred.item())                           # [1,1,1]
        context = torch.cat((context[0,1:,0], x[0,1,:])).unsqueeze(-1).unsqueeze(0)   # [1,S,1]
        x = torch.cat((x[0,1:,0],pred)).unsqueeze(-1).unsqueeze(0)                                 # [1,L,1]

    return list_pred # [n]


# Generate error of n_predictions vs target future values

def error_pred(func, list_pred_x, list_pred):
    return torch.abs(torch.tensor(list_pred) - func(torch.tensor(list_pred_x))).tolist()


# Generate lists of predicted values and corresponding x_axis

list_pred = prediction_n_step(n=num_pred_steps, input_dec=input_dec, input_enc=input_enc)
list_pred_x = [seq_x[-1].item()+(i)*step for i in range(num_pred_steps)]

abs_error = error_pred(func=func, list_pred_x=list_pred_x, list_pred=list_pred)


# Plots

fig, ax = plt.subplots(1,2, figsize=(14,7))

ax[0].plot(seq_x[600:], sequence[600:], 'r', label='train')
ax[0].plot(list_pred_x, list_pred, 'b.-', label='pred.')
ax[0].plot(list_pred_x, func(torch.tensor(list_pred_x)).tolist(),'g.-', label='target')
ax[0].set_title('Prediction of ' + str(num_pred_steps) + ' steps' )
ax[0].set_xlabel('Domain [t]')
ax[0].set_ylabel('Func')
ax[0].legend()

ax[1].plot(list_pred_x, abs_error, '.--')
ax[1].set_title('Abs. Error of Predictions' )
ax[1].set_xlabel('Domain [t]')
ax[1].set_ylabel('Abs. Error')



# Plot learning rate 

plt.figure()
x = torch.linspace(1,1000,1000)
y = torch.min(x**(-1),x*n_warmup_steps**(-1.5))
plt.plot(x,y)

# Plot target

plt.figure()
x = torch.linspace(0,4*torch.pi,1000)
y = func(x)
plt.plot(x,y)


