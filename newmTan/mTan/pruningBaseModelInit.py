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
import models
import utils
import sys
import pdb
from pruneGru import prune_standard_gru, prune_hh_standard_gru, prune_bi_gru, prune_hh_gru

# torch.autograd.set_detect_anomaly(True)
warnings.filterwarnings("ignore")

log_file = open("pruningBaseInit.csv", mode="w", newline='')
log_file_2 = open("pruningBaseInitBest.csv", mode="w", newline='')
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
    
    rec_gru_component = rec.gru_rnn
    dec_gru_component = dec.gru_rnn
    classifier_gru_component = classifier.gru_rnn
    
    encoder_prune_time_start = time.time()
    if pruning_iteration<=2:
        rec_gru_component = prune_bi_gru(rec_gru_component)
        rec_gru_component = prune_hh_gru(rec_gru_component)

    encoder_prune_time = time.time() - encoder_prune_time_start
    # print(f"Encoder prune time: {encoder_prune_time:.4f}s")
    if pruning_iteration <= 1:
        dec_gru_component = prune_bi_gru(dec_gru_component)
        dec_gru_component = prune_hh_gru(dec_gru_component)
   
    classifier_prune_time_start = time.time()
    if pruning_iteration <= 2:
        classifier_gru_component = prune_standard_gru(classifier_gru_component) 
        classifier_gru_component = prune_hh_standard_gru(classifier_gru_component)
    classifier_prune_time = time.time() - classifier_prune_time_start

    print("=========================================================")

    ih_l0 = classifier.gru_rnn.weight_ih_l0.data
    index = ih_l0.shape[0] // 3
    # if pruning_iteration <= 1:
    classifier._initialize_numberHidden(index)
        # classifier._initialize_classifier()
    classifier.update_input_size(index, pruning_iteration)
    classifier.gru_rnn.hidden_size = index

    index = rec.gru_rnn.weight_ih_l0.data.shape[0] // 3
    if pruning_iteration<=2:
        rec._initialize_nhidden(index)
        rec.gru_rnn.hidden_size = index
        rec.gru_rnn.hidden_size = index
        # rec._initialize_hiddens_to_z0()
        rec.update_input_size(index, pruning_iteration)
    if pruning_iteration <= 1:
        decoder_index = dec.gru_rnn.weight_ih_l0.data.shape[0]//3
        dec._initialize_nhidden(decoder_index)
        dec.gru_rnn.hidden_size = decoder_index
        dec.gru_rnn.hidden_size = decoder_index
        dec.prune_decoder_layers()

    # dec.update_input_size(decoder_index)

    print("=========================================================")
   
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


        torch.save(rec, f"modelsBaseInit/{pruning_iteration}_rec.pth")
        torch.save(dec, f"modelsBaseInit/{pruning_iteration}_dec.pth")
        torch.save(classifier, f"modelsBaseInit/{pruning_iteration}_classifier.pth")
        torch.save(model, f"modelsBaseInit/{pruning_iteration}_model.pth")

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
    # print(model)
    # model = torch.load("SecondCopiedModel.pt")
    params = (list(rec.parameters()) + list(dec.parameters()) + list(classifier.parameters()))
    # model = PN.loadPAIModel(model, 'best_model_pai.pt').to('cuda')
    # sys.exit()
    # rec = model.rec
    # dec = model.dec
    # classifier = model.classifier

    # torch.save(rec, "rec.pth")
    # torch.save(dec, "dec.pth")
    # torch.save(classifier, "classifier.pth")
    # sys.exit()
    # print(model)

    # rec = torch.load("1_rec.pth")
    # dec = torch.load("1_dec.pth")
    # classifier = torch.load("1_classifier.pth")
    
    
    params = (list(rec.parameters()) + list(dec.parameters()) + list(classifier.parameters()))
    print('Before loading parameters:', utils.count_parameters(rec), utils.count_parameters(dec), utils.count_parameters(classifier))
    
    if(args.justTest):
        
        # model.eval()
        rec.eval()
        dec.eval()
        classifier.eval()
        test_loss, test_acc, test_auc = utils.evaluate_classifier(
            rec, test_loader, args=args, classifier=classifier, reconst=True, num_sample=1, dim=dim, device=device)
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
    
    # rec, dec, classifier = train_step(rec, dec, classifier, train_loader, optimizer, criterion)

    rec, dec, classifier = prune(rec, dec, classifier, train_loader, optimizer, criterion)
    