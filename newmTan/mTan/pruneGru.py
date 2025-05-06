import torch
import torch.nn as nn
import warnings



    
def prune_standard_gru(gru_component):

    layers = []
    for name, layer in gru_component.named_parameters():
        layers.append(str(name))
    

    ih_l0 = gru_component.weight_ih_l0.data
    ih_l0_bias = gru_component.bias_ih_l0.data

    hh_l0 = gru_component.weight_hh_l0.data
    hh_l0_bias = gru_component.bias_hh_l0.data

    index = ih_l0.shape[0]//3
    print("Shape of normal gru is: ", ih_l0.shape[0])
    print("We have index size as: ", index)

    W_ir = ih_l0[:index, :]  # Reset gate weights
    W_iz = ih_l0[index:index*2, :]  # Update gate weights
    W_in = ih_l0[index*2:, :]  # New gate weights

    l1_norm_ir = W_ir.abs().sum(dim=1)  # L1 norm for reset gate
    l1_norm_iz = W_iz.abs().sum(dim=1)  # L1 norm for update gate
    l1_norm_in = W_in.abs().sum(dim=1)  # L1 norm for new gate

    threshold = 0.97
    # threshold = 0.9
    num_units_to_keep = int(threshold * index)  # 80% of 256
    keep_indices_ir = torch.topk(l1_norm_ir, num_units_to_keep).indices
    keep_indices_iz = torch.topk(l1_norm_iz, num_units_to_keep).indices
    keep_indices_in = torch.topk(l1_norm_in, num_units_to_keep).indices

    keep_indices_ir = keep_indices_ir.sort().values
    keep_indices_iz = keep_indices_iz.sort().values
    keep_indices_in = keep_indices_in.sort().values

    W_ir_pruned = W_ir[keep_indices_ir]
    W_iz_pruned = W_iz[keep_indices_iz]
    W_in_pruned = W_in[keep_indices_in]

    pruned_weight_ih = torch.cat((W_ir_pruned, W_iz_pruned, W_in_pruned), dim=0)

    with torch.no_grad():
        gru_component.weight_ih_l0.data = pruned_weight_ih.clone().detach()

    # ih bias
    W_ir = ih_l0_bias[:index]  # Reset gate weights
    W_iz = ih_l0_bias[index:index*2]  # Update gate weights
    W_in = ih_l0_bias[index*2:]  # New gate weights

    keep_indices_ir = keep_indices_ir.sort().values
    keep_indices_iz = keep_indices_iz.sort().values
    keep_indices_in = keep_indices_in.sort().values

    W_ir_pruned = W_ir[keep_indices_ir]
    W_iz_pruned = W_iz[keep_indices_iz]
    W_in_pruned = W_in[keep_indices_in]

    pruned_weight_ih = torch.cat((W_ir_pruned, W_iz_pruned, W_in_pruned), dim=0)
    
    with torch.no_grad():
        gru_component.bias_ih_l0.data = pruned_weight_ih.clone().detach()

    # hh
    W_ir = hh_l0[:index, :]  # Reset gate weights
    W_iz = hh_l0[index:index*2, :]  # Update gate weights
    W_in = hh_l0[index*2:, :]  # New gate weights

    l1_norm_ir = W_ir.abs().sum(dim=1)  # L1 norm for reset gate
    l1_norm_iz = W_iz.abs().sum(dim=1)  # L1 norm for update gate
    l1_norm_in = W_in.abs().sum(dim=1)  # L1 norm for new gate

    num_units_to_keep = int(threshold * index)
    keep_indices_ir = torch.topk(l1_norm_ir, num_units_to_keep).indices
    keep_indices_iz = torch.topk(l1_norm_iz, num_units_to_keep).indices
    keep_indices_in = torch.topk(l1_norm_in, num_units_to_keep).indices

    keep_indices_ir = keep_indices_ir.sort().values
    keep_indices_iz = keep_indices_iz.sort().values
    keep_indices_in = keep_indices_in.sort().values

    W_ir_pruned = W_ir[keep_indices_ir]
    W_iz_pruned = W_iz[keep_indices_iz]
    W_in_pruned = W_in[keep_indices_in]

    pruned_weight_ih = torch.cat((W_ir_pruned, W_iz_pruned, W_in_pruned), dim=0)

    with torch.no_grad():
        gru_component.weight_hh_l0.data = pruned_weight_ih.clone().detach()

    # hh bias
    W_ir = hh_l0_bias[:index]  # Reset gate weights
    W_iz = hh_l0_bias[index:index*2]  # Update gate weights
    W_in = hh_l0_bias[index*2:]  # New gate weights

    keep_indices_ir = keep_indices_ir.sort().values
    keep_indices_iz = keep_indices_iz.sort().values
    keep_indices_in = keep_indices_in.sort().values

    W_ir_pruned = W_ir[keep_indices_ir]
    W_iz_pruned = W_iz[keep_indices_iz]
    W_in_pruned = W_in[keep_indices_in]

    pruned_weight_ih = torch.cat((W_ir_pruned, W_iz_pruned, W_in_pruned), dim=0)

    with torch.no_grad():
        gru_component.bias_hh_l0.data = pruned_weight_ih.clone().detach()

    
    
    return gru_component


def prune_bi_gru(gru_component):

    layers = []
    for name, layer in gru_component.named_parameters():
        layers.append(str(name))
    
    

    ih_l0 = gru_component.weight_ih_l0.data
    ih_l0_bias = gru_component.bias_ih_l0.data

    hh_l0 = gru_component.weight_hh_l0.data
    hh_l0_bias = gru_component.bias_hh_l0.data

    if "weight_ih_l0_reverse" in layers:
        ih_l0_reverse = gru_component.weight_ih_l0_reverse.data
        ih_l0_reverse_bias = gru_component.bias_ih_l0_reverse.data

        hh_l0_reverse = gru_component.weight_hh_l0_reverse.data
        hh_l0_reverse_bias = gru_component.bias_hh_l0_reverse.data

    index = ih_l0.shape[0]//3

  
    W_ir = ih_l0[:index, :]  # Reset gate weights
    W_iz = ih_l0[index:index*2, :]  # Update gate weights
    W_in = ih_l0[index*2:, :]  # New gate weights

    l1_norm_ir = W_ir.abs().sum(dim=1)  # L1 norm for reset gate
    l1_norm_iz = W_iz.abs().sum(dim=1)  # L1 norm for update gate
    l1_norm_in = W_in.abs().sum(dim=1)  # L1 norm for new gate

    threshold = 0.97
    num_units_to_keep = int(threshold * index)  # 80% of 256
    keep_indices_ir = torch.topk(l1_norm_ir, num_units_to_keep).indices
    keep_indices_iz = torch.topk(l1_norm_iz, num_units_to_keep).indices
    keep_indices_in = torch.topk(l1_norm_in, num_units_to_keep).indices

    keep_indices_ir = keep_indices_ir.sort().values
    keep_indices_iz = keep_indices_iz.sort().values
    keep_indices_in = keep_indices_in.sort().values

    W_ir_pruned = W_ir[keep_indices_ir]
    W_iz_pruned = W_iz[keep_indices_iz]
    W_in_pruned = W_in[keep_indices_in]

    pruned_weight_ih = torch.cat((W_ir_pruned, W_iz_pruned, W_in_pruned), dim=0)

    gru_component.weight_ih_l0.data = pruned_weight_ih.clone().detach()

    # ih bias
    W_ir = ih_l0_bias[:index]  # Reset gate weights
    W_iz = ih_l0_bias[index:index*2]  # Update gate weights
    W_in = ih_l0_bias[index*2:]  # New gate weights

    keep_indices_ir = keep_indices_ir.sort().values
    keep_indices_iz = keep_indices_iz.sort().values
    keep_indices_in = keep_indices_in.sort().values

    W_ir_pruned = W_ir[keep_indices_ir]
    W_iz_pruned = W_iz[keep_indices_iz]
    W_in_pruned = W_in[keep_indices_in]

    pruned_weight_ih = torch.cat((W_ir_pruned, W_iz_pruned, W_in_pruned), dim=0)

    gru_component.bias_ih_l0.data = pruned_weight_ih.clone().detach()

    # hh
    W_ir = hh_l0[:index, :]  # Reset gate weights
    W_iz = hh_l0[index:index*2, :]  # Update gate weights
    W_in = hh_l0[index*2:, :]  # New gate weights

    l1_norm_ir = W_ir.abs().sum(dim=1)  # L1 norm for reset gate
    l1_norm_iz = W_iz.abs().sum(dim=1)  # L1 norm for update gate
    l1_norm_in = W_in.abs().sum(dim=1)  # L1 norm for new gate

    num_units_to_keep = int(threshold * index)
    keep_indices_ir = torch.topk(l1_norm_ir, num_units_to_keep).indices
    keep_indices_iz = torch.topk(l1_norm_iz, num_units_to_keep).indices
    keep_indices_in = torch.topk(l1_norm_in, num_units_to_keep).indices

    keep_indices_ir = keep_indices_ir.sort().values
    keep_indices_iz = keep_indices_iz.sort().values
    keep_indices_in = keep_indices_in.sort().values

    W_ir_pruned = W_ir[keep_indices_ir]
    W_iz_pruned = W_iz[keep_indices_iz]
    W_in_pruned = W_in[keep_indices_in]

    pruned_weight_ih = torch.cat((W_ir_pruned, W_iz_pruned, W_in_pruned), dim=0)

    gru_component.weight_hh_l0.data = pruned_weight_ih.clone().detach()

    # hh bias
    W_ir = hh_l0_bias[:index]  # Reset gate weights
    W_iz = hh_l0_bias[index:index*2]  # Update gate weights
    W_in = hh_l0_bias[index*2:]  # New gate weights

    keep_indices_ir = keep_indices_ir.sort().values
    keep_indices_iz = keep_indices_iz.sort().values
    keep_indices_in = keep_indices_in.sort().values

    W_ir_pruned = W_ir[keep_indices_ir]
    W_iz_pruned = W_iz[keep_indices_iz]
    W_in_pruned = W_in[keep_indices_in]

    pruned_weight_ih = torch.cat((W_ir_pruned, W_iz_pruned, W_in_pruned), dim=0)

    gru_component.bias_hh_l0.data = pruned_weight_ih.clone().detach()

    new_index = ih_l0_reverse.shape[0]//3

    gru_component.hidden_size = new_index

    if "weight_ih_l0_reverse" in layers:

        W_ir = ih_l0_reverse[:index, :]  # Reset gate weights
        W_iz = ih_l0_reverse[index:index*2, :]  # Update gate weights
        W_in = ih_l0_reverse[index*2:, :]  # New gate weights

        l1_norm_ir = W_ir.abs().sum(dim=1)  # L1 norm for reset gate
        l1_norm_iz = W_iz.abs().sum(dim=1)  # L1 norm for update gate
        l1_norm_in = W_in.abs().sum(dim=1)  # L1 norm for new gate

        threshold = 0.97
        num_units_to_keep = int(threshold * index)  # 80% of 256
        keep_indices_ir = torch.topk(l1_norm_ir, num_units_to_keep).indices
        keep_indices_iz = torch.topk(l1_norm_iz, num_units_to_keep).indices
        keep_indices_in = torch.topk(l1_norm_in, num_units_to_keep).indices

        keep_indices_ir = keep_indices_ir.sort().values
        keep_indices_iz = keep_indices_iz.sort().values
        keep_indices_in = keep_indices_in.sort().values

        W_ir_pruned = W_ir[keep_indices_ir]
        W_iz_pruned = W_iz[keep_indices_iz]
        W_in_pruned = W_in[keep_indices_in]

        pruned_weight_ih = torch.cat((W_ir_pruned, W_iz_pruned, W_in_pruned), dim=0)

        gru_component.weight_ih_l0_reverse.data = pruned_weight_ih.clone().detach()

        W_ir = ih_l0_reverse_bias[:index]  # Reset gate weights
        W_iz = ih_l0_reverse_bias[index:index*2]  # Update gate weights
        W_in = ih_l0_reverse_bias[index*2:]  # New gate weights

        keep_indices_ir = keep_indices_ir.sort().values
        keep_indices_iz = keep_indices_iz.sort().values
        keep_indices_in = keep_indices_in.sort().values

        W_ir_pruned = W_ir[keep_indices_ir]
        W_iz_pruned = W_iz[keep_indices_iz]
        W_in_pruned = W_in[keep_indices_in]

        pruned_weight_ih = torch.cat((W_ir_pruned, W_iz_pruned, W_in_pruned), dim=0)

        gru_component.bias_ih_l0_reverse.data = pruned_weight_ih.clone().detach()

        W_ir = hh_l0_reverse[:index, :]  # Reset gate weights
        W_iz = hh_l0_reverse[index:index*2, :]  # Update gate weights
        W_in = hh_l0_reverse[index*2:, :]  # New gate weights

        l1_norm_ir = W_ir.abs().sum(dim=1)  # L1 norm for reset gate
        l1_norm_iz = W_iz.abs().sum(dim=1)  # L1 norm for update gate
        l1_norm_in = W_in.abs().sum(dim=1)  # L1 norm for new gate

        threshold = 0.97
        num_units_to_keep = int(threshold * index)  # 80% of 256
        keep_indices_ir = torch.topk(l1_norm_ir, num_units_to_keep).indices
        keep_indices_iz = torch.topk(l1_norm_iz, num_units_to_keep).indices
        keep_indices_in = torch.topk(l1_norm_in, num_units_to_keep).indices

        keep_indices_ir = keep_indices_ir.sort().values
        keep_indices_iz = keep_indices_iz.sort().values
        keep_indices_in = keep_indices_in.sort().values

        W_ir_pruned = W_ir[keep_indices_ir]
        W_iz_pruned = W_iz[keep_indices_iz]
        W_in_pruned = W_in[keep_indices_in]

        pruned_weight_ih = torch.cat((W_ir_pruned, W_iz_pruned, W_in_pruned), dim=0)

        gru_component.weight_hh_l0_reverse.data = pruned_weight_ih.clone().detach()

        W_ir = hh_l0_reverse_bias[:index]  # Reset gate weights
        W_iz = hh_l0_reverse_bias[index:index*2]  # Update gate weights
        W_in = hh_l0_reverse_bias[index*2:]  # New gate weights

        keep_indices_ir = keep_indices_ir.sort().values
        keep_indices_iz = keep_indices_iz.sort().values
        keep_indices_in = keep_indices_in.sort().values

        W_ir_pruned = W_ir[keep_indices_ir]
        W_iz_pruned = W_iz[keep_indices_iz]
        W_in_pruned = W_in[keep_indices_in]

        pruned_weight_ih = torch.cat((W_ir_pruned, W_iz_pruned, W_in_pruned), dim=0)

        gru_component.bias_hh_l0_reverse.data = pruned_weight_ih.clone().detach()

        new_index = ih_l0_reverse.shape[0]//3

    for name, param in gru_component.named_parameters():
        print(f"{name} : {param.shape}")

    
    
    return gru_component

def prune_hh_gru(gru_component):

    layers = []
    for name, layer in gru_component.named_parameters():
        layers.append(str(name))

    hh_l0 = gru_component.weight_hh_l0.data

    if "weight_ih_l0_reverse" in layers:
        hh_l0_reverse = gru_component.weight_hh_l0_reverse.data

    index = hh_l0.shape[1]
    # print(f"Index: ",index)

    threshold = 0.97
    num_units_to_keep = int(threshold * index) 
    
    # hh
    W_ir = hh_l0[:index, :]  # Reset gate weights
    W_iz = hh_l0[index:index*2, :]  # Update gate weights
    W_in = hh_l0[index*2:, :]  # New gate weights

    l1_norm_ir = W_ir.abs().sum(dim=0)  # L1 norm for reset gate
    l1_norm_iz = W_iz.abs().sum(dim=0)  # L1 norm for update gate
    l1_norm_in = W_in.abs().sum(dim=0)  # L1 norm for new gate

   
    num_units_to_keep = int(threshold * index)
    keep_indices_ir = torch.topk(l1_norm_ir, num_units_to_keep).indices
    keep_indices_iz = torch.topk(l1_norm_iz, num_units_to_keep).indices
    keep_indices_in = torch.topk(l1_norm_in, num_units_to_keep).indices

    keep_indices_ir = keep_indices_ir.sort().values
    keep_indices_iz = keep_indices_iz.sort().values
    keep_indices_in = keep_indices_in.sort().values

    W_ir_pruned = W_ir[:,keep_indices_ir]
    W_iz_pruned = W_iz[:,keep_indices_iz]
    W_in_pruned = W_in[:,keep_indices_in]

    pruned_weight_hh = torch.cat((W_ir_pruned, W_iz_pruned, W_in_pruned), dim=0)

    gru_component.weight_hh_l0.data = pruned_weight_hh.clone().detach()

    if "weight_ih_l0_reverse" in layers:

        W_ir = hh_l0_reverse[:index, :]  # Reset gate weights
        W_iz = hh_l0_reverse[index:index*2, :]  # Update gate weights
        W_in = hh_l0_reverse[index*2:, :]  # New gate weights

        l1_norm_ir = W_ir.abs().sum(dim=0)  # L1 norm for reset gate
        l1_norm_iz = W_iz.abs().sum(dim=0)  # L1 norm for update gate
        l1_norm_in = W_in.abs().sum(dim=0)  # L1 norm for new gate

        threshold = 0.97
        num_units_to_keep = int(threshold * index)  
        keep_indices_ir = torch.topk(l1_norm_ir, num_units_to_keep).indices
        keep_indices_iz = torch.topk(l1_norm_iz, num_units_to_keep).indices
        keep_indices_in = torch.topk(l1_norm_in, num_units_to_keep).indices

        keep_indices_ir = keep_indices_ir.sort().values
        keep_indices_iz = keep_indices_iz.sort().values
        keep_indices_in = keep_indices_in.sort().values

        W_ir_pruned = W_ir[:,keep_indices_ir]
        W_iz_pruned = W_iz[:,keep_indices_iz]
        W_in_pruned = W_in[:,keep_indices_in]

        pruned_weight_ih = torch.cat((W_ir_pruned, W_iz_pruned, W_in_pruned), dim=0)

        gru_component.weight_hh_l0_reverse.data = pruned_weight_ih.clone().detach()
    
    return gru_component

def prune_hh_standard_gru(gru_component):

    hh_l0 = gru_component.weight_hh_l0.data

    index = hh_l0.shape[1]
    print("Shape of hidden gru unit: ", hh_l0.shape[1])
    

    threshold = 0.97
    num_units_to_keep = int(threshold * index) 
    print(f"Index: ",num_units_to_keep)
    # hh
    W_ir = hh_l0[:index, :]  # Reset gate weights
    W_iz = hh_l0[index:index*2, :]  # Update gate weights
    W_in = hh_l0[index*2:, :]  # New gate weights

    l1_norm_ir = W_ir.abs().sum(dim=0)  # L1 norm for reset gate
    l1_norm_iz = W_iz.abs().sum(dim=0)  # L1 norm for update gate
    l1_norm_in = W_in.abs().sum(dim=0)  # L1 norm for new gate

    num_units_to_keep = int(threshold * index)
    keep_indices_ir = torch.topk(l1_norm_ir, num_units_to_keep).indices
    keep_indices_iz = torch.topk(l1_norm_iz, num_units_to_keep).indices
    keep_indices_in = torch.topk(l1_norm_in, num_units_to_keep).indices

    keep_indices_ir = keep_indices_ir.sort().values
    keep_indices_iz = keep_indices_iz.sort().values
    keep_indices_in = keep_indices_in.sort().values

    W_ir_pruned = W_ir[:,keep_indices_ir]
    W_iz_pruned = W_iz[:,keep_indices_iz]
    W_in_pruned = W_in[:,keep_indices_in]

    pruned_weight_hh = torch.cat((W_ir_pruned, W_iz_pruned, W_in_pruned), dim=0)

    with torch.no_grad():
        gru_component.weight_hh_l0.data = pruned_weight_hh.clone().detach()

 
    return gru_component

if __name__ == "__main__":

    input_size = 256
    hidden_size = 128
    seq_length = 278
    batch_size = 50

    bi_gru = BidirectionalGRU(input_size, hidden_size)

    X = torch.randn(batch_size, seq_length, input_size)

    print("Named Parameters of Bidirectional GRU:")
    for name, param in bi_gru.named_parameters():
        print(f"{name}: {param.shape}")

    H = bi_gru(X)
    print("\nConcatenated Hidden States (Forward + Backward):\n", H.shape)



