import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import csv
from perforatedai import pb_network as PN
from perforatedai import pb_globals as PBG
import warnings
from random import SystemRandom
import modelsPAI as models
import utils
import sys
import pdb
from pruneGru import prune_standard_gru, prune_hh_standard_gru, prune_bi_gru, prune_hh_gru
from prune_buffers import prune_buffer

# torch.autograd.set_detect_anomaly(True)
warnings.filterwarnings("ignore")

log_file = open("pruningTrained.csv", mode="w", newline='')
# log_file = open("pruning_at_initialisation_2.csv", mode="w", newline='')
log_file_2 = open("pruningTrainedBest.csv", mode="w", newline='')
# log_file_2 = open("pruning_at_initialisation_2_best.csv", mode="w", newline='')
log_writer = csv.writer(log_file)
log_writer_2 = csv.writer(log_file_2)
log_writer.writerow(["Iteration", "Epoch", "Recon_Loss", "CE_Loss", "Accuracy", "MSE", "Val_Loss", "Val_Acc", "Test_Acc", "Test_AUC"])
log_writer_2.writerow(["Iteration", "Epoch", "Val_Acc", "Val_AUC"])


pruning_iteration = 0
current_test_auc = 0
best_test_auc = 0

parser = argparse.ArgumentParser()
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--std', type=float, default=0.01)
parser.add_argument('--latent-dim', type=int, default=32)
parser.add_argument('--rec-hidden', type=int, default=32)
parser.add_argument('--gen-hidden', type=int, default=50)
parser.add_argument('--embed-time', type=int, default=128)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--enc', type=str, default='mtan_rnn')
parser.add_argument('--dec', type=str, default='mtan_rnn')
parser.add_argument('--fname', type=str, default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--justTest', type=int, default=0)
parser.add_argument('--split', type=int, default=0)
parser.add_argument('--n', type=int, default=8000)
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--quantization', type=float, default=0.1, 
                    help="Quantization on the physionet dataset.")
parser.add_argument('--classif', action='store_true', 
                    help="Include binary classification loss")
parser.add_argument('--freq', type=float, default=10.)
parser.add_argument('--k-iwae', type=int, default=10)
parser.add_argument('--norm', action='store_true')
parser.add_argument('--kl', action='store_true')
parser.add_argument('--learn-emb', action='store_true')
parser.add_argument('--dataset', type=str, default='physionet')
parser.add_argument('--alpha', type=int, default=100.)
parser.add_argument('--old-split', type=int, default=1)
parser.add_argument('--nonormalize', action='store_true')
parser.add_argument('--enc-num-heads', type=int, default=1)
parser.add_argument('--dec-num-heads', type=int, default=1)
parser.add_argument('--num-ref-points', type=int, default=128)
parser.add_argument('--classify-pertp', action='store_true')
parser.add_argument('--multiplier', type=float, default=1)

args = parser.parse_args()

class fullModel(nn.Module):
    def __init__(self, rec, dec, classifier):
        super(fullModel, self).__init__()
        self.rec = rec
        self.dec = dec
        self.classifier = classifier


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
        
    #for post processsing its just getting passed the output, which is (h_t,c_t). Then it wants to just pass along h_t as the output for the function to be passed to the parents while retaining both
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

PBG.inputDimensions = [-1, -1, 0]

PBG.modulesToConvert.append(models.mtan_time_embedder)
PBG.modulesToConvert.append(models.multiTimeAttention)
PBG.modulesToConvert.append(nn.GRU)
PBG.modulesToConvert.append(models.reverseGru)

PBG.modulesWithProcessing.append(nn.GRU)
PBG.moduleProcessingClasses.append(GRUCellProcessor)
PBG.modulesWithProcessing.append(models.reverseGru)
PBG.moduleProcessingClasses.append(ReverseGRUCellProcessor)

def train_step(rec, dec, classifier, train_loader, optimizer, criterion):

    global pruning_iteration, best_test_auc, current_test_auc
    
    best_rec, best_dec, best_classifier = 0, 0, 0

    pruning_iteration +=1
    best_test_auc = float('-inf')
    best_test_acc = float("-inf")
    best_rec = None
    best_dec = None
    best_classifier = None
    total_time = 0.
    iteration = 0
    best_val_loss = float('inf')
    total_time = 0.

    params = (list(rec.parameters()) + list(dec.parameters()) + list(classifier.parameters()))
    print('parameters:', utils.count_parameters(rec), utils.count_parameters(dec), utils.count_parameters(classifier))
    # print('paramCount:', utils.count_parameters(rec)+ utils.count_parameters(dec)+ utils.count_parameters(classifier))
    optimizer.state_dict().clear()
    optimizer = optim.Adam(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # l2_norm = calculate_l2_norm(rec, dec, classifier)
    for itr in range(1, args.niters + 1):
        train_recon_loss, train_ce_loss = 0, 0
        mse = 0
        train_n = 0
        train_acc = 0
        #avg_reconst, avg_kl, mse = 0, 0, 0
        if args.kl:
            wait_until_kl_inc = 10
            if itr < wait_until_kl_inc:
                kl_coef = 0.
            else:
                kl_coef = (1-0.99** (itr - wait_until_kl_inc))
        else:
            kl_coef = 1

        for train_batch, label in train_loader:

            
            train_batch, label = train_batch.to(device), label.to(device)
            batch_len  = train_batch.shape[0]
            
            observed_data = train_batch[:, :, :dim]
            observed_mask = train_batch[:, :, dim:2*dim]
            observed_tp = train_batch[:, :, -1]

            encoder_input = (torch.cat((observed_data, observed_mask), 2), observed_mask)
            
            out = rec(torch.cat((observed_data, observed_mask), 2), observed_tp)

            qz0_mean = out[:, :, :args.latent_dim]
            qz0_logvar = out[:, :, args.latent_dim:]
            
            epsilon = torch.randn(args.k_iwae, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]).to(device)
            
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
            z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])

            # print(rec)
            pred_y = classifier(z0)

            pred_x = dec(z0, observed_tp[None, :, :].repeat(args.k_iwae, 1, 1).view(-1, observed_tp.shape[1]))
            pred_x = pred_x.view(args.k_iwae, batch_len, pred_x.shape[1], pred_x.shape[2]) #nsample, batch, seqlen, dim
            # compute loss

            logpx, analytic_kl = utils.compute_losses(dim, train_batch, qz0_mean, qz0_logvar, pred_x, args, device)
            recon_loss = -(torch.logsumexp(logpx - kl_coef * analytic_kl, dim=0).mean(0) - np.log(args.k_iwae))
            label = label.unsqueeze(0).repeat_interleave(args.k_iwae, 0).view(-1)
          
            ce_loss = criterion(pred_y, label)
            loss = recon_loss + args.alpha*ce_loss 
            loss = recon_loss + args.alpha*ce_loss
            
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()

            train_ce_loss += ce_loss.item() * batch_len
            train_recon_loss += recon_loss.item() * batch_len
            train_acc += (pred_y.argmax(1) == label).sum().item()/args.k_iwae
            train_n += batch_len
            mse += utils.mean_squared_error(observed_data, pred_x.mean(0), observed_mask) * batch_len
   
        val_loss, val_acc, val_auc = utils.evaluate_classifier(rec, val_loader, args=args, classifier=classifier, reconst=True, num_sample=1, dim=dim)

        test_loss, test_acc, test_auc = utils.evaluate_classifier(rec, test_loader, args=args, classifier=classifier, reconst=True, num_sample=1, dim=dim)
        

        log_writer.writerow([itr, train_recon_loss / train_n, train_ce_loss / train_n, train_acc / train_n, mse / train_n, val_loss, val_acc, test_acc, test_auc])
        print('Pruning Iter: {}, Iter: {}, recon_loss: {:.4f}, ce_loss: {:.4f}, acc: {:.4f}, mse: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, val_auc: {:.4f}, test_acc: {:.4f}, test_auc: {:.4f}'
              .format(pruning_iteration, itr, train_recon_loss/train_n, train_ce_loss/train_n, 
                      train_acc/train_n, mse/train_n, val_loss, val_acc, val_auc, test_acc, test_auc))

        if test_auc >= best_test_auc:
            iteration = itr
            best_test_auc = test_auc
            best_test_acc = test_acc
            best_rec = rec
            best_dec = dec
            best_classifier = classifier
            # optimizer_state_dict = optimizer.state_dict()
    log_writer_2.writerow([pruning_iteration, iteration, best_test_acc, best_test_auc])
    print("Pruning iteration: ", pruning_iteration)
    print("Best auc: ", best_test_auc)
    return best_rec, best_dec, best_classifier
    # return rec, dec, classifier

def prune_step(rec, dec, classifier):
    global pruning_iteration
    
    for i in range(len(rec.gru_rnn.layerArray)):
        
        rec_gru_component = rec.gru_rnn.layerArray[i]
        dec_gru_component = dec.gru_rnn.layerArray[i]
        classifier_gru_component = classifier.gru_rnn.layerArray[i]
    
        if args.multiplier == 0.0625 and pruning_iteration <= 2:
            print("Encoder GRU's getting pruned here")
            rec_gru_component = prune_bi_gru(rec_gru_component)
            rec_gru_component = prune_hh_gru(rec_gru_component)

       
        print("Decoder GRU's getting pruned here")
        if args.multiplier == 0.0625 and pruning_iteration <= 1:
            if pruning_iteration < 5:
                dec_gru_component = prune_bi_gru(dec_gru_component)
                dec_gru_component = prune_hh_gru(dec_gru_component)

        if args.multiplier == 0.0625 and pruning_iteration <= 2:
            classifier_gru_component = prune_standard_gru(classifier_gru_component) 
            classifier_gru_component = prune_hh_standard_gru(classifier_gru_component)

    
        rec.gru_rnn.layerArray[i] = rec_gru_component
        dec.gru_rnn.layerArray[i] = dec_gru_component
        classifier.gru_rnn.layerArray[i] = classifier_gru_component
    
    print("=========================================================")

    for i in range(len(rec.gru_rnn.layerArray)):
        start_init = time.time()
        ih_l0 = classifier.gru_rnn.layerArray[i].weight_ih_l0.data
        index = ih_l0.shape[0] // 3
        
        if i == 0:
            classifier._initialize_numberHidden(index)
            # classifier._initialize_classifier()
            classifier.update_input_size(index, pruning_iteration)
        classifier.gru_rnn.layerArray[i].hidden_size = index

        index = rec.gru_rnn.layerArray[i].weight_ih_l0.data.shape[0] // 3
        
        if args.multiplier == 0.0625 and pruning_iteration <= 2:
            if i == 0:
                rec._initialize_nhidden(index)
                rec.update_input_size(index, pruning_iteration)

            rec.gru_rnn.layerArray[i].hidden_size = index
            rec.gru_rnn.layerArray[i].hidden_size = index
        # if args.multiplier == 0.0625 and pruning_iteration <= 1:
        # rec.prune_encoder_layers()
        # rec._initialize_hiddens_to_z0()
        
        
        
        
        decoder_index = dec.gru_rnn.layerArray[i].weight_ih_l0.data.shape[0]//3
        if args.multiplier == 0.0625 and pruning_iteration <= 1:
            if i == 0:
                dec._initialize_nhidden(decoder_index)
                dec.prune_decoder_layers()
        if args.multiplier == 0.0625 and pruning_iteration <= 1:
            dec.gru_rnn.layerArray[i].hidden_size = decoder_index
            dec.gru_rnn.layerArray[i].hidden_size = decoder_index
        

    # dec.update_input_size(decoder_index)

    print("=========================================================")
   

    if args.multiplier == 0.0625 and pruning_iteration <= 2:
        for submodule_name, submodule in rec.named_modules():
                    if "gru_rnn" in submodule_name:
                        for name2, buffer in submodule.named_buffers():
                            if submodule:
                                if "skipWeights" in name2:
                                    # print(name)
                                    # print(name2)

                                    size = rec.hiddens_to_z0[0].layerArray[0].weight.shape[1]
                                        # print("New size: ", size)
                                    new_buffer = prune_buffer(buffer, number = size)
                                        # print("New buffer shape: ", new_buffer.shape)

                                    with torch.no_grad():
                                        submodule.register_buffer(name2, new_buffer)
    
    if pruning_iteration < 5 and (args.multiplier == 0.0625 and pruning_iteration <= 1):
        for submodule_name, submodule in dec.named_modules():
                    if "gru_rnn" in submodule_name:
                        for name2, buffer in submodule.named_buffers():
                            if submodule:
                                if "skipWeights" in name2:
                                    # print(name)
                                    # print(name2)
                                    
                                    new_buffer = prune_buffer(buffer)
                                    with torch.no_grad():
                                        submodule.register_buffer(name2, new_buffer)
    if args.multiplier == 0.0625 and pruning_iteration <= 2:
        for submodule_name, submodule in classifier.named_modules():
                    if "gru_rnn" in submodule_name:
                        for name2, buffer in submodule.named_buffers():
                            if submodule:
                                if "skipWeights" in name2:
                                    # print("Old classifier size buffer is: ", buffer.shape)          
                                    # if buffer.shape[2] == 24:
                                        # new_buffer = prune_buffer(buffer, threshold = 0.05, number = 23)
                                    # else:
                                        # new_buffer = prune_buffer(buffer)
                                    new_buffer = prune_buffer(buffer)
                                    # print("New classifier size buffer is: ", new_buffer.shape)
                                    with torch.no_grad():
                                        submodule.register_buffer(name2, new_buffer)
       
    if args.multiplier == 0.0625 and pruning_iteration <= 1:
        
        for submodule_name, submodule in rec.named_modules():
                    if "hiddens_to_z0" in submodule_name:
                        for name2, buffer in submodule.named_buffers():
                            if submodule:
                                if "skipWeights" == name2:

                                    if buffer.shape[2] != 4:
                                        if args.multiplier == 0.0625 and buffer.shape[2] == 6:
                                            continue
                                        # print("Old Buffer Shape: ",buffer.shape)
                                        new_buffer = prune_buffer(buffer)
                                        # print("New Buffer Shape: ",new_buffer.shape)
                                        with torch.no_grad():
                                            submodule.register_buffer(name2, new_buffer)

    if args.multiplier == 0.0625 and pruning_iteration <= 1:
        unitNum = 0
        for submodule_name, submodule in dec.named_modules():
                    if "z0_to_obs" in submodule_name:
                        for name2, buffer in submodule.named_buffers():
                            if submodule:
                                if "skipWeights" == name2:
                                    # print(f"Unit name is over here: ", name2)
                                    # print("Shape is: ", buffer.shape)


                                    # if buffer.shape[2] != 4 and buffer.shape[2] != 41:
                                    if buffer.shape[2] != 41 or unitNum == 0:    
                                        new_buffer = prune_buffer(buffer)
                                        # print("The new buffer shape is: ", new_buffer.shape)
                                        unitNum += 1
                                        with torch.no_grad():
                                            submodule.register_buffer(name2, new_buffer)
    
    for submodule_name, submodule in classifier.named_modules():
                    for name2, buffer in submodule.named_buffers():
                        if submodule:
                            if "skipWeights" == name2 and "gru_rnn" not in submodule_name:
                                
                                if buffer.shape[2] != 2:
                                    new_buffer = prune_buffer(buffer)
                                    with torch.no_grad():
                                        submodule.register_buffer(name2, new_buffer)

    if pruning_iteration < 5 and (args.multiplier == 0.0625 and pruning_iteration <= 1):
        for submodule_name, submodule in dec.named_modules():
            if "gru_rnn" in submodule_name:
                for name2, buffer in submodule.named_buffers():
                    if submodule:
                        if "skipWeights" in name2:
                            
                            new_buffer = prune_buffer(buffer)
                            with torch.no_grad():
                                submodule.register_buffer(name2, new_buffer)

    return rec, dec, classifier

def prune(rec, dec, classifier, train_loader, optimizer, criterion):
    global pruning_iteration, current_test_auc, best_test_auc
    
    while True:

        rec, dec, classifier = prune_step(rec, dec, classifier)
    
        print("Pruning iteration is: ", pruning_iteration)
        rec.zero_grad()
        dec.zero_grad()
        classifier.zero_grad()
        torch.cuda.empty_cache()

        # PBG.modulesToConvert = []
        # PBG.modulesToConvert.append(models.mtan_time_embedder)
        # PBG.modulesToConvert.append(models.multiTimeAttention)
        # PBG.modulesToConvert.append(nn.GRU)
        # PBG.modulesToConvert.append(models.reverseGru)

        # PBG.modluesWithProcessing = []
        # PBG.moduleProcessingClasses = []

        # PBG.modluesWithProcessing.append(nn.GRU)
        # PBG.moduleProcessingClasses.append(GRUCellProcessor)
        # PBG.modluesWithProcessing.append(models.reverseGru)
        # PBG.moduleProcessingClasses.append(ReverseGRUCellProcessor)

        # print(rec.gru_rnn.processor.clear_processor())
        # print(dec.gru_rnn.processor.clear_processor())
        rec.gru_rnn.processor.clear_processor()
        dec.gru_rnn.processor.clear_processor()
        # print(rec.gru_rnn.processor.h_t_d)
        # print(rec.gru_rnn.processor.c_t_d)
        # print(decoder.gru_rnn.processor)
        # print(rec.gru_rnn.processor)
        # print(rec)
        # print(dec)
        # print(classifier)
        
        print("Encoder")
        print(rec)
        print("Decodder")
        print(dec)
        print("Classifier")
        print(classifier)
        optimizer = optim.Adam(params, lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        
        
        new_rec, new_dec, new_classifier = train_step(rec, dec, classifier, train_loader, optimizer, criterion)
        
        rec = new_rec
        dec = new_dec
        classifier = new_classifier


        torch.save(rec, f"modelsTrained/{pruning_iteration}_rec.pth")
        torch.save(dec, f"modelsTrained/{pruning_iteration}_dec.pth")
        torch.save(classifier, f"modelsTrained/{pruning_iteration}_classifier.pth")

        # del rec
        # del dec
        # del classifier

    return rec, dec, classifier

if __name__ == '__main__':
    experiment_id = int(SystemRandom().random()*100000)
    print(args, experiment_id)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print("The device being used here is: ", device)
    
    if args.dataset == 'physionet':
        data_obj = utils.get_physionet_data(args, 'cpu', args.quantization)
    elif args.dataset == 'mimiciii':
        data_obj = utils.get_mimiciii_data(args)
    
    train_loader = data_obj["train_dataloader"]
    test_loader = data_obj["test_dataloader"]
    val_loader = data_obj["val_dataloader"]
    dim = data_obj["input_dim"]
    
    internal = int(100 * args.multiplier)
    nlin = int(50 * args.multiplier)
    embed_time = 128
    embed_time = int(embed_time * args.multiplier)
    args.latent_dim = int(args.latent_dim * args.multiplier)
    args.rec_hidden = int(args.rec_hidden * args.multiplier)
    args.gen_hidden = int(args.gen_hidden * args.multiplier)
        
    
    if args.enc == 'enc_rnn3':
        rec = models.enc_rnn3(
            dim, torch.linspace(0, 1., embed_time), args.latent_dim, args.rec_hidden, embed_time=embed_time , internal=internal, nlin=nlin, learn_emb=args.learn_emb).to(device)
    elif args.enc == 'mtan_rnn':
        rec = models.enc_mtan_rnn(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, args.rec_hidden, 
            embed_time=embed_time,nlin=nlin, learn_emb=args.learn_emb, num_heads=args.enc_num_heads, device=device).to(device)

    if args.dec == 'rnn3':
        dec = models.dec_rnn3(
            dim, torch.linspace(0, 1., embed_time), args.latent_dim, args.gen_hidden, embed_time=embed_time, nlin=nlin, learn_emb=args.learn_emb).to(device)
    elif args.dec == 'mtan_rnn':
        dec = models.dec_mtan_rnn(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, args.gen_hidden, 
            embed_time=embed_time, nlin=nlin, learn_emb=args.learn_emb, num_heads=args.dec_num_heads, device=device).to(device)
    
    classifier = models.create_classifier(args.latent_dim, internal, args.rec_hidden).to(device)
    
    model = fullModel(rec, dec, classifier)
    model = PN.loadPAIModel(model, 'smallest.pt').to('cuda')
    print(model)
    # sys.exit()
    # torch.save(model, "CheckingModel.pt")
    params = (list(model.rec.parameters()) + list(model.dec.parameters()) + list(model.classifier.parameters()))
    print('parameters:', utils.count_parameters(rec), utils.count_parameters(dec), utils.count_parameters(classifier))
    print(model)
    if(args.justTest):
        model.eval()
        test_loss, test_acc, test_auc = utils.evaluate_classifier(
            model.rec, test_loader, args=args, classifier=model.classifier, reconst=True, num_sample=1, dim=dim, device=device)
        print('test_acc')
        print(test_acc)
        print('test_auc')
        print(test_auc)
        
        
        
        
        import pdb; pdb.set_trace()
        exit(0)    


    params = (list(rec.parameters()) + list(dec.parameters()) + list(classifier.parameters()))
    print('parameters:', utils.count_parameters(rec), utils.count_parameters(dec), utils.count_parameters(classifier))

    optimizer = optim.Adam(params, lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    print(rec)
    print(dec)
    print(classifier)

    rec, dec, classifier = prune(rec, dec, classifier, train_loader, optimizer, criterion)


