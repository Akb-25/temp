import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class reverseGru(nn.GRU):
    def __init__(self, *args, **kwargs):
        super(reverseGru, self).__init__(*args, **kwargs)


class create_classifier(nn.Module):
 
    def __init__(self, latent_dim, internal = 300, nhidden=16, N=2):
        super(create_classifier, self).__init__()
        self.gru_rnn = reverseGru(latent_dim, nhidden, batch_first=True)
        self.numberHidden = nhidden
        self.N = N
        self.classifier = nn.Sequential(
            nn.Linear(self.numberHidden, internal),
            nn.ReLU(),
            nn.Linear(internal, internal),
            nn.ReLU(),
            nn.Linear(internal, N))

    def _initialize_numberHidden(self, n):
        self.numberHidden = n
        
    def prune_classifier(self, threshold=0.03):
        

        original_weights_1 = self.classifier[0].weight.data.clone()
        original_biases_1 = self.classifier[0].bias.data.clone()
        l1_norms_1 = torch.sum(torch.abs(original_weights_1), dim=1)  

        top_k_1 = int((1 - threshold) * l1_norms_1.size(0))
        _, top_indices_1 = torch.topk(l1_norms_1, top_k_1)
            
        new_layer_1 = nn.Linear(self.numberHidden, top_k_1).to(device)
            
        with torch.no_grad():
            new_layer_1.weight.data = original_weights_1[top_indices_1, :]
            new_layer_1.bias.data = original_biases_1[top_indices_1]
        self.classifier[0] = new_layer_1
            
        original_weights_2 = self.classifier[2].weight.data.clone()
        original_biases_2 = self.classifier[2].bias.data.clone()
            
        l1_norms_2 = torch.sum(torch.abs(original_weights_2), dim=1) 
        top_k_2_out = int((1 - threshold) * l1_norms_2.size(0))
        _, top_indices_2_out = torch.topk(l1_norms_2, top_k_2_out)
            
        l1_norms_2_in = torch.sum(torch.abs(original_weights_2), dim=0)
        top_k_2_in = int((1 - threshold) * l1_norms_2_in.size(0))
        _, top_indices_2_in = torch.topk(l1_norms_2_in, top_k_2_in)
            
        new_layer_2 = nn.Linear(top_k_2_in, top_k_2_out).to(device)
            
        with torch.no_grad():
            new_layer_2.weight.data = original_weights_2[top_indices_2_out][:, top_indices_2_in]
            new_layer_2.bias.data = original_biases_2[top_indices_2_out]
        self.classifier[2] = new_layer_2
            
        original_weights_3 = self.classifier[4].weight.data.clone()
        original_biases_3 = self.classifier[4].bias.data.clone()
            
        l1_norms_3 = torch.sum(torch.abs(original_weights_3), dim=0)  
        # print(f"Third layer l1 norms shape: ",l1_norms_3.shape)
        top_k_3_in = int((1 - threshold) * l1_norms_3.size(0))
        _, top_indices_3_in = torch.topk(l1_norms_3, top_k_3_in)
            
        new_layer_3 = nn.Linear(top_k_3_in, self.N).to(device)
            
        with torch.no_grad():
            new_layer_3.weight.data = original_weights_3[:, top_indices_3_in]
            new_layer_3.bias.data = original_biases_3
        self.classifier[4] = new_layer_3
        
    def update_input_size(self, new_input_size, pruning_iteration):

        original_weights = self.classifier[0].weight.data.clone()
        
        original_biases = self.classifier[0].bias.data.clone()

        # original_biases = original_biases[0].layerArray[i].bias.data.clone()     
        
        l1_norms = torch.sum(torch.abs(original_weights), dim=0)   
        
        _, top_indices = torch.topk(l1_norms, new_input_size)      
        
        self.numberHidden = new_input_size
        
        new_input_layer = nn.Linear(self.numberHidden, 12)

        with torch.no_grad():
            new_input_layer.weight.data = original_weights[:, top_indices] 
            new_input_layer.bias.data = original_biases  
            
            
        self.classifier[0] = new_input_layer
        if pruning_iteration < 10:
            self.prune_classifier()
        
    def forward(self, z):
        _, out = self.gru_rnn(z)
        return self.classifier(out.squeeze(0))
        # return self.classifier(out)
    

class multiTimeAttention(nn.Module):
    
    def __init__(self, input_dim, nhidden=16, 
                 embed_time=16, num_heads=1):
        super(multiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.top_k_ratio = 0.97
        self.temp = 0
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time), 
                                      nn.Linear(embed_time, embed_time),
                                      nn.Linear(input_dim*num_heads, nhidden)])
        
        
    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)
        p_attn = F.softmax(scores, dim = -2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn*value.unsqueeze(-3), -2), p_attn
    
    def reinitialize_dim(self, d):
        # print(" HAPPENING NOW ----------------------------------------------")
        self.dim = d
        new_input_to_last = self.dim * self.h

       
        original_weights = self.linears[-1].weight.data.clone()     
        original_biases = self.linears[-1].bias.data.clone() 

        l1_norms = torch.sum(torch.abs(original_weights), dim=0)
        _, top_indices = torch.topk(l1_norms, self.dim*self.h) 

        new_layer = nn.Linear(self.dim*self.h, self.nhidden)

        with torch.no_grad():
            new_layer.weight.data = original_weights[:,top_indices]
            new_layer.bias.data = original_biases

        self.linears[-1] = new_layer     

    def forward(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        batch, seq_len, dim = value.size()
        if mask is not None:
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        
        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        
        # query_layer = self.linears[0](query)  
        # key_layer = self.linears[1](key)     

        # query_reshaped = query_layer.view(query.size(0), -1, self.h, self.embed_time_k)
        # key_reshaped = key_layer.view(key.size(0), -1, self.h, self.embed_time_k)

        # query_transposed = query_reshaped.transpose(1, 2)
        # key_transposed = key_reshaped.transpose(1, 2)
       
        # query, key = query_transposed, key_transposed
        
        x, _ = self.attention(query, key, value, mask, dropout)
        

        x = x.transpose(1, 2).contiguous() \
             .view(batch, -1, self.h * dim)
        
        new_dim = int(x.shape[2])
        if new_dim != 82 and new_dim != self.temp:
            self.reinitialize_dim(new_dim)
            self.temp = new_dim
            
        output = self.linears[-1](x)
        
        return output
    
    

    
class mtan_time_embedder(nn.Module):
    def __init__(self, device, embed_time):
        super(mtan_time_embedder, self).__init__()
        self.device = device
        self.periodic = nn.Linear(1, embed_time-1)
        self.linear = nn.Linear(1, 1)
        
    def forward(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    
class enc_mtan_rnn(nn.Module):
    def __init__(self, input_dim, query, latent_dim=2, nhidden=16, nlin = 50,
                 embed_time=16, num_heads=1, learn_emb=False, device='cuda'):
        super(enc_mtan_rnn, self).__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.learn_emb = learn_emb
        self.att = multiTimeAttention(2*input_dim, nhidden, embed_time, num_heads)
        self.gru_rnn = nn.GRU(nhidden, nhidden, bidirectional = True, batch_first=True)
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(2*nhidden, nlin),
            nn.ReLU(),
            nn.Linear(nlin, latent_dim * 2))
        if learn_emb:
            self.embedder1 = mtan_time_embedder(self.device, embed_time)
            self.embedder2 = mtan_time_embedder(self.device, embed_time)
    
    def _initialize_nhidden(self, n):
        self.nhidden = n

    def _initialize_hiddens_to_z0(self):
        """Reinitialize the hiddens_to_z0 layer."""
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(2 * self.nhidden, 50),
            nn.ReLU(),
            nn.Linear(50, 20 * 2)
        ).to(device)
    def update_input_size(self, new_input_size):
        
        original_weights = self.hiddens_to_z0[0].weight.data.clone()  
        original_biases = self.hiddens_to_z0[0].bias.data.clone()     
        
        l1_norms = torch.sum(torch.abs(original_weights), dim=0)   
        
        _, top_indices = torch.topk(l1_norms, 2*new_input_size)      
        
        self.nhidden = new_input_size
        
        new_input_layer = nn.Linear(2 * self.nhidden, 50)
        # with torch.no_grad:
        new_input_layer.weight.data = original_weights[:, top_indices] 
        new_input_layer.bias.data = original_biases  
            
        self.hiddens_to_z0[0] = new_input_layer
        
        self.prune_encoder_layers()
        # print(self.hiddens_to_z0)


    def prune_encoder_layers(self, threshold=0.03):

        original_weights_1 = self.hiddens_to_z0[0].weight.data.clone()
        original_biases_1 = self.hiddens_to_z0[0].bias.data.clone()
        l1_norms_1 = torch.sum(torch.abs(original_weights_1), dim=1)
        top_k_1 = int((1 - threshold) * l1_norms_1.size(0))
        _, top_indices_1 = torch.topk(l1_norms_1, top_k_1)
            
        new_layer_1 = nn.Linear(original_weights_1.size(1), top_k_1).to(self.device)

        with torch.no_grad():
            new_layer_1.weight.data = original_weights_1[top_indices_1, :]
            new_layer_1.bias.data = original_biases_1[top_indices_1]
        self.hiddens_to_z0[0] = new_layer_1

        original_weights_2 = self.hiddens_to_z0[2].weight.data.clone()
        original_biases_2 = self.hiddens_to_z0[2].bias.data.clone()
        l1_norms_2 = torch.sum(torch.abs(original_weights_2), dim=0)
        top_k_2 = int((1 - threshold) * l1_norms_2.size(0))
        _, top_indices_2 = torch.topk(l1_norms_2, top_k_2)
            
        new_layer_2 = nn.Linear(top_k_2, original_weights_2.size(0)).to(self.device)
        with torch.no_grad():
            new_layer_2.weight.data = original_weights_2[:, top_indices_2]
            new_layer_2.bias.data = original_biases_2
        self.hiddens_to_z0[2] = new_layer_2
    
    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
    
    def fixed_time_embedding(self, pos):
        d_model=self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    # def print_versions(self, name, **tensors):
    #     for tensor_name, tensor in tensors.items():
    #         if tensor is not None:
    #             print(f"{name} - {tensor_name} version: {tensor._version}")
   
    def forward(self, x, time_steps):
        #time_steps = time_steps.cpu()
        mask = x[:, :, self.dim:]
        mask = torch.cat((mask, mask), 2)
        self.query = self.query.to(self.device)
        if self.learn_emb:
            key = self.embedder1(time_steps).to(self.device)
            query = self.embedder2(self.query.unsqueeze(0)).to(self.device)
            
        else:
            key = self.fixed_time_embedding(time_steps).to(self.device)
            query = self.fixed_time_embedding(self.query.unsqueeze(0)).to(self.device)
        # self.print_versions("Before Encoder GRU x: ", x=x)
        out = self.att(query, key, x, mask)
        # self.print_versions("Before Encoder GRU: ", out=out)
        out, _ = self.gru_rnn(out)
        # self.print_versions("After Encoder GRU: ",out = out)
        out = self.hiddens_to_z0(out)
        return out
    
   

   
class dec_mtan_rnn(nn.Module):
 
    def __init__(self, input_dim, query, latent_dim=2, nhidden=16, nlin=50, 
                 embed_time=16, num_heads=1, learn_emb=False, device='cuda'):
        super(dec_mtan_rnn, self).__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.learn_emb = learn_emb
        self.att = multiTimeAttention(2*nhidden, 2*nhidden, embed_time, num_heads)
        self.gru_rnn = nn.GRU(latent_dim, nhidden, bidirectional=True, batch_first =True)    
        self.z0_to_obs = nn.Sequential(
            nn.Linear(2*nhidden, nlin),
            nn.ReLU(),
            nn.Linear(nlin, input_dim))
        if learn_emb:
            self.embedder1 = mtan_time_embedder(self.device, embed_time)
            self.embedder2 = mtan_time_embedder(self.device, embed_time)
        
    def _initialize_nhidden(self, n):
        # print("Hidden in decoder values is: ",self.nhidden)
        self.nhidden = n

    def _initialize_z0_to_obs(self):
        self.z0_to_obs = nn.Sequential(
            nn.Linear(2*self.nhidden, 50),
            nn.ReLU(),
            nn.Linear(50, self.dim))

    def update_input_size(self, new_input_size):
       
        original_weights = self.z0_to_obs[0].weight.data.clone()  
        original_biases = self.z0_to_obs[0].bias.data.clone()     
        
        l1_norms = torch.sum(torch.abs(original_weights), dim=0)   
        
        _, top_indices = torch.topk(l1_norms, 2*new_input_size)      
        
        self.nhidden = new_input_size
        
        new_input_layer = nn.Linear(2 * self.nhidden, 50)

        with torch.no_grad():    
            new_input_layer.weight.data = original_weights[:, top_indices] 
            new_input_layer.bias.data = original_biases  
            
        self.z0_to_obs[0] = new_input_layer


    def prune_decoder_layers(self, threshold=0.03):
        # Prune first layer's output in z0_to_obs
        original_weights_1 = self.z0_to_obs[0].weight.data.clone()
        original_biases_1 = self.z0_to_obs[0].bias.data.clone()
        l1_norms_1 = torch.sum(torch.abs(original_weights_1), dim=1)
        top_k_1 = int((1 - threshold) * l1_norms_1.size(0))
        _, top_indices_1 = torch.topk(l1_norms_1, top_k_1)
            
        new_layer_1 = nn.Linear(original_weights_1.size(1), top_k_1).to(self.device)
        
        with torch.no_grad():
            new_layer_1.weight.data = original_weights_1[top_indices_1, :]
            new_layer_1.bias.data = original_biases_1[top_indices_1]
        self.z0_to_obs[0] = new_layer_1

        # Prune second layer's input in z0_to_obs
        original_weights_2 = self.z0_to_obs[2].weight.data.clone()
        original_biases_2 = self.z0_to_obs[2].bias.data.clone()
        l1_norms_2 = torch.sum(torch.abs(original_weights_2), dim=0)
        top_k_2 = int((1 - threshold) * l1_norms_2.size(0))
        _, top_indices_2 = torch.topk(l1_norms_2, top_k_2)
            
        new_layer_2 = nn.Linear(top_k_2, original_weights_2.size(0)).to(self.device)
        with torch.no_grad():
            new_layer_2.weight.data = original_weights_2[:, top_indices_2]
            new_layer_2.bias.data = original_biases_2
        self.z0_to_obs[2] = new_layer_2

        # print(self.z0_to_obs)
        # self.prune_decoder_layers()

    
    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
        
        
    def fixed_time_embedding(self, pos):
        d_model = self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    
    # def print_versions(self, name, **tensors):
    #     for tensor_name, tensor in tensors.items():
    #         if tensor is not None:
    #             print(f"{name} - {tensor_name} version: {tensor._version}")

    def forward(self, z, time_steps):
        # self.print_versions("Decoder GRU", gru = self.gru_rnn)
        self.query = self.query.to(self.device)
        # self.print_versions("Decoder Before GRU", z=z)
        out, _ = self.gru_rnn(z)
        # self.print_versions("Decoder After GRU", out=out)

        # out, _ = self.gru_rnn(z)
        #time_steps = time_steps.cpu()
        if self.learn_emb:
            query = self.embedder1(time_steps).to(self.device)
            key = self.embedder2(self.query.unsqueeze(0)).to(self.device)
        else:
            query = self.fixed_time_embedding(time_steps).to(self.device)
            key = self.fixed_time_embedding(self.query.unsqueeze(0)).to(self.device)
        out = self.att(query, key, out)
        out = self.z0_to_obs(out) 
        return out
        