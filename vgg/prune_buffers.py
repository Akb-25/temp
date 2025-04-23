import torch
import torch.nn as nn

# torch.set_printoptions(threshold = torch.inf)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class GRUCellProcessor():
    #Post processing does eventually need to return h_t and c__t, but h_t gets modified py the PB
    #nodes first so it needs to be extracted in post 1, and then gets added back in post 2
    def post_n1(self, *args, **kawrgs):
        h_t = args[0][0]
        c_t = args[0][1]
        #store the cell state temporarily and just use the hidden state to do PB functions
        self.c_t_n = c_t
        return h_t
    def post_n2(self, *args, **kawrgs):
        h_t = args[0]
        return h_t, self.c_t_n
    #these Grus are just getting passed input and no hidden state for some reason so just pass it along
    def pre_d(self, *args, **kwargs):
        return args, kwargs
        
    #for post processsing its just getting passed the output, which is (h_t,c_t). Then it wants to just pass along h_t as the output for the function to be passed to the parent while retaining both
    def post_d(self, *args, **kawrgs):
        h_t = args[0][0]
        c_t = args[0][1]
        self.h_t_d = h_t
        self.c_t_d = c_t
        return h_t
    def clear_processor(self):
        for var in ['c_t_n', 'h_t_d', 'c_t_d']:
            if(not getattr(self,var,None) is None):
                delattr(self,var)
    
#the classifier gru is passing along the cell state instead of the hidden state so use that isntead
class ReverseGRUCellProcessor():
    #Post processing does eventually need to return h_t and c__t, but c_t gets modified py the PB
    #nodes first so it needs to be extracted in post 1, and then gets added back in post 2
    def post_n1(self, *args, **kawrgs):
        h_t = args[0][0]
        c_t = args[0][1]
        #store the cell state temporarily and just use the hidden state to do PB functions
        self.h_t_n = h_t
        return c_t
    def post_n2(self, *args, **kawrgs):
        c_t = args[0]
        return self.h_t_n, c_t
    #these Grus are just getting passed input and no hidden state for some reason so just pass it along
    def pre_d(self, *args, **kwargs):
        return args, kwargs
        
    #for post processsing its just getting passed the output, which is (h_t,c_t). Then it wants to just pass along h_t as the output for the function to be passed to the parent while retaining both
    def post_d(self, *args, **kawrgs):
        h_t = args[0][0]
        c_t = args[0][1]
        self.h_t_d = h_t
        self.c_t_d = c_t
        return c_t
    def clear_processor(self):
        for var in ['h_t_n', 'h_t_d', 'c_t_d']:
            if(not getattr(self,var,None) is None):
                delattr(self,var)
class fullModel(nn.Module):
    def __init__(self, rec, dec, classifier):
        super(fullModel, self).__init__()
        self.rec = rec
        self.dec = dec
        self.classifier = classifier
        
    def forward(self, observed_data, observed_mask, observed_tp):
        #import pdb; pdb.set_trace()
        out = self.rec(torch.cat((observed_data, observed_mask), 2), observed_tp)
        qz0_mean, qz0_logvar = out[:, :, :args.latent_dim], out[:, :, args.latent_dim:]
        epsilon = torch.randn(args.k_iwae, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]).to(device)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
        z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
        pred_y = self.classifier(z0)
        pred_x = self.dec(
            z0, observed_tp[None, :, :].repeat(args.k_iwae, 1, 1).view(-1, observed_tp.shape[1]))
        pred_x = pred_x.view(args.k_iwae, batch_len, pred_x.shape[1], pred_x.shape[2]) #nsample, batch, seqlen, dim
        return pred_x, pred_y, qz0_mean, qz0_logvar

# def prune_buffer(tensor, threshold=0.05):
    
#     # Prune output
#     print(tensor.shape)
#     l1_norms = torch.sum(torch.abs(tensor), dim=1) 
#     print("Shape of l1 norms are: ", l1_norms.shape)
#     print("Size of l1 norms are: ", l1_norms.size(1))
#     top_k = int((1 - threshold) * l1_norms.size(1))

#     _, top_indices = torch.topk(l1_norms, top_k, dim=1)
#     print("Top indices shape is here: ", top_indices.shape)

#     pruned_tensor = torch.zeros(tensor.size(0), tensor.size(1), top_k).to(tensor.device)
#     print("New tensor shape is here: ", pruned_tensor.shape)
#     # for i in range(tensor.size(1)):  # Iterate over the second dimension (N)
#     #     pruned_tensor[:, :, i] = 
#         # pruned_tensor[:, i, :] = tensor[:, i, :].gather(1, top_indices[:, i].unsqueeze(-1).expand(-1, -1, tensor.size(2)))
#     print(top_indices)
#     for i in range(tensor.size(0)):  # Iterate over the batch dimension (B)
#         print("pruned_tensor", pruned_tensor.shape)
#         print("tensor ", tensor[i].shape)
#         print("top indices: ", top_indices[i].shape)
#         pruned_tensor[i] = tensor[i, :, top_indices[i]]
#     return pruned_tensor
def prune_buffer(tensor, threshold=0.03, number=0):
    
   
    l1_norms = torch.sum(torch.abs(tensor), dim=1) 
    if number:
        top_k = number
    else:
        top_k = int((1 - threshold) * l1_norms.size(1))
    # print("L1 norms: ", l1_norms.shape)
    
    _, top_indices = torch.topk(l1_norms, top_k, dim=1, largest=True, sorted=False)

    pruned_tensor = torch.zeros(tensor.size(0), tensor.size(1), top_k).to(tensor.device)
    
    for i in range(tensor.size(0)): 
        # with torch.no_grad:
        pruned_tensor[i] = nn.Parameter(tensor[i, :, top_indices[i]])
        pruned_tensor = pruned_tensor.detach()
    return pruned_tensor

if __name__ == "__main__":

    # input_tensor = torch.randn(1, 64)

    # pruned_tensor = prune_buffer(tensor = input_tensor)

    # print("Original Tensor:")
    # print(input_tensor)
    # print("\nPruned Tensor:")
    # print(pruned_tensor)
    # print("Pruned tensor shape: ", pruned_tensor.shape)
    # print(type(pruned_tensor))


    model = torch.load("SecondCopiedModel.pt")
    print(model)
    
    dec = model.dec

    # for name, module in dec.named_modules():
    #     if name:
    #         for submodule_name, submodule in module.named_modules():
    #             for name2, buffer in submodule.named_buffers():
    #                 if submodule:
    #                     if "skipWeights" in name2:
    #                         print(submodule_name)
    #                         print(name2)
    #                         print(buffer.shape)

    for submodule_name, submodule in dec.named_modules():
        if "gru_rnn" in submodule_name:
            for name2, buffer in submodule.named_buffers():
                if submodule:
                    if "skipWeights" in name2:
                        print(submodule_name)
                        print(name2)
                        print(buffer.shape)
                        new_buffer = prune_buffer(buffer)
                        submodule.register_buffer(name2, new_buffer)
                        print(new_buffer.shape)


    for name, module in model.named_modules():
        if name:
            for submodule_name, submodule in module.named_modules():
                if "hiddens_to_z0" in submodule_name:
                    for name2, buffer in submodule.named_buffers():
                        if submodule:
                            if "skipWeights" == name2:
                                print(name2)
                                print(submodule_name)
                                print(buffer.shape)
                                print(buffer.shape[2])
                                if buffer.shape[2] != 4:
                                    new_buffer = prune_buffer(buffer)
                                    submodule.register_buffer(name2, new_buffer)
                                    print(new_buffer.shape)

    for name, module in model.named_modules():
        if name:
            for submodule_name, submodule in module.named_modules():
                # print(submodule_name)
                if "z0_to_obs" in submodule_name:
                    for name2, buffer in submodule.named_buffers():
                        if submodule:
                            if "skipWeights" == name2:
                                print(name2)
                                print(submodule_name)
                                print(buffer.shape)
                                print(buffer.shape[2])
                                if buffer.shape[2] != 4 and buffer.shape[2] != 41:
                                    new_buffer = prune_buffer(buffer)
                                    submodule.register_buffer(name2, new_buffer)
                                    print(new_buffer.shape)
    sys.exit()
    for name, module in model.named_modules():
        if name:
            for submodule_name, submodule in module.named_modules():
                # print(submodule_name)
                if "classifier" in submodule_name:
                    for name2, buffer in submodule.named_buffers():
                        if submodule:
                            if "skipWeights" == name2:
                                print(name2)
                                print(submodule_name)
                                print(buffer.shape)
                                print(buffer.shape[2])
                                if buffer.shape[2] != 2:
                                    new_buffer = prune_buffer(buffer)
                                    submodule.register_buffer(name2, new_buffer)
                                    print(new_buffer.shape)
                #         # print(submodule)
                #         print(name2)
                #         print(buffer.shape)
                #         print(submodule)
    
    for name, module in model.named_modules():
        if name:
            for submodule_name, submodule in module.named_modules():
                if "gru_rnn" in submodule_name:
                    for name2, buffer in submodule.named_buffers():
                        if submodule:
                            if "skipWeights" in name2:
                                print(submodule_name)
                                print(name)
                                print(name2)
                                print(buffer.shape)
                                new_buffer = prune_buffer(buffer)
                                
                                submodule.register_buffer(name2, new_buffer)
                                print(new_buffer.shape)
    for name, module in model.named_modules():
        if name:
            for name2, buffer in module.named_buffers():
                if "gru" in name2 and "skipWeights" in name2:
                    print(name)
                    print(name2)
                    print(buffer.shape)
                    print(buffer)
    


    for name, module in model.named_modules():
        if name:
            for submodule_name, submodule in module.named_modules():
                if "gru_rnn" in submodule_name:
                    for name2, buffer in submodule.named_buffers():
                        if submodule:
                            if "skipWeights" in name2:
                                print(submodule_name)
                                print(name2)
                                
                                new_buffer = prune_buffer(buffer)
                                
                                submodule.register_buffer(name2, new_buffer)

    for name, module in model.named_modules():
        if name:
            for name2, buffer in module.named_buffers():
                if "gru" in name2 and "skipWeights" in name2:
                    print(name)
                    print(name2)
                    print(buffer.shape)
                    print(buffer)
    
        # for name, buffer in model.named_buffers():
        #     if "gru" in name and "skipWeights" in name:            
                    # print("Old buffer shape: ",buffer.shape)
                # print(buffer)
    #             new_buffer = buffer
    #             new_buffer = prune_buffer(new_buffer)
    #             buffer = new_buffer
    #             print("New buffer shape in main: ", buffer.shape)
    #             print(buffer)
    #             print(name, buffer.shape)
    #             model.register_buffer(name, new_buffer)