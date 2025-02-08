# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from scipy.stats import rankdata
from collections import OrderedDict

from networks.pgl_sum.pgl_sum import PGL_SUM
from networks.csta.csta import set_model

from model.utils.evaluation_metrics import evaluate_summary, evaluate_knapsack_opt
from model.utils.generate_summary import generate_summary, get_gt
from model.utils.evaluate_map import generate_mrsum_seg_scores, top50_summary, top15_summary

from model.utils.early_stopping import EarlyStopping
from matplotlib import pyplot as plt
import json
from copy import deepcopy

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
            
def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

class Solver(object):
    def __init__(self, config=None, train_loader=None, val_loader=None, test_loader=None, ckpt_path=None):
        
        self.model, self.optimizer, self.writer, self.scheduler = None, None, None, None

        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.global_step = 0

        self.criterion = nn.MSELoss(reduction='none').to(self.config.device)
        self.ckpt_path = ckpt_path
        
        if config.p_uncond > 0:
            self.null_video = np.load('dataset/null_video.npy')

    def build(self):
        """ Define your own summarization model here """
        
        if self.config.model == 'PGL_SUM':
            self.model = PGL_SUM(input_size=self.config.input_dim, output_size=1024, num_segments=4, heads=8, fusion="add", pos_enc="absolute")
            self.model.to(self.config.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.l2_reg)
            # self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
            self.scheduler = None
            self.init_weights(self.model, init_type='xavier')
        
        elif self.config.model == 'CSTA':
            self.model = set_model('GoogleNet_Attention', None, 'TD', None, 'FPE', 'TD', 'PGL_SUM', True, 0.6, True,
                                    True, 'Final', 'kv', 'KC', True, input_dim=self.config.input_dim, batch_size=self.config.batch_size)
            self.model.to(self.config.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.l2_reg)
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.config.gamma)
        
        else:
            print("Wrong model")
            exit()

        self.early_stopper = EarlyStopping()
        self.writer = SummaryWriter(log_dir=f'/home/yejin/data/projects/yejin/VideoSum/CSTA/tensorboard/{self.config.tag}')
  
        if self.ckpt_path != None:
            print("Loading Model: ", self.ckpt_path)
            self.model.load_state_dict(torch.load(self.ckpt_path))
            self.model.to(self.config.device)

    def log_gpu_memory(self, step):
        allocated = torch.cuda.memory_allocated(self.config.device) / (1024 ** 2)  # MB 단위
        reserved = torch.cuda.memory_reserved(self.config.device) / (1024 ** 2)  # MB 단위
        self.writer.add_scalar("GPU/Allocated_MB", allocated, step)
        self.writer.add_scalar("GPU/Reserved_MB", reserved, step)


    def train(self):
        best_f1score = -1.0
        best_map50 = -1.0
        best_map15 = -1.0
        best_srho = -1.0
        best_ktau = -1.0
        best_f1score_epoch = 0
        best_map50_epoch = 0
        best_map15_epoch = 0
        best_srho_epoch = 0
        best_ktau_epoch = 0
        
        if self.config.ema == True:
            self.ema_model = deepcopy(self.model).to(self.config.device)
            requires_grad(self.ema_model, False)
            update_ema(self.ema_model, self.model, 0)
            self.ema_model.eval()
     
        for epoch_i in range(self.config.epochs):
            print("[Epoch: {0:6}]".format(str(epoch_i)+"/"+str(self.config.epochs)))
            self.model.train()

            loss_history = []
            num_batches = int(len(self.train_loader))
            iterator = iter(self.train_loader)

            for batch_idx in tqdm(range(num_batches)):

                self.optimizer.zero_grad()
                data = next(iterator)

                frame_features = data['features'].to(self.config.device)
                gtscore = data['gtscore'].to(self.config.device)
                mask = data['mask'].to(self.config.device) #torch.Size([batch_size, max_len])
                n_frames = data['n_frames']
                gt_summary = data['gt_summary']

                #print(f"batched frames: {frame_features.shape}", flush=True) #torch.Size([batch_size, frame_num, emb_dim])
                
                if self.config.clamp:
                    gtscore = torch.clamp(gtscore, 0.05, 0.95)
                
                if self.config.model == 'CSTA':
                    frame_features = frame_features.unsqueeze(1) #torch.Size([batch_size, channel=1, frame_num, emb_dim])
                    #print(f"unsqueezed frame features: {frame_features.shape}", flush=True)
                    frame_features = frame_features.expand(-1, 3, -1, -1) 
                    #print(f"frame features: {frame_features.shape}", flush=True) #torch.Size([batch_size, channel=3, frame_num, emb_dim])
                    #print(f"gtscore: {gtscore.shape}, {gtscore.dtype}", flush=True) #torch.Size([batch_size, frame_num])
                    
                    score = self.model(frame_features, mask)
                    #print(f"score: {score.sum(dim=1)}", flush=True)
                    loss = self.criterion(score, gtscore)

                    #print(f"loss: {loss.shape}", flush=True) #torch.Size([batch_size, frame_num])
                    loss = loss.mean() 
                    #print(f"mean loss: {loss.shape}", flush=True) #torch.Size([])
                    self.writer.add_scalar('loss/train', loss, epoch_i)
                
                else:
                    score, weights = self.model(frame_features, mask)
                    #print(f"pgl_sum score: {score.shape}", flush=True) # [batch_size, frame_num]
                    loss = self.criterion(score[mask], gtscore[mask]).mean()

                loss.backward()
                loss_history.append(loss.item())
                #print(f"train batch loss: {loss.item()}", flush=True)

                if self.config.model == 'PGL_SUM':
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()
                if self.config.ema == True:
                    update_ema(self.ema_model, self.model, self.config.ema_decay)
                    
                if self.config.individual == True and batch_idx + 1 != num_batches:
                    if self.config.dataset == 'summe':
                        val_f1_tmp_list = []
                        val_map50_tmp_list = []
                        val_map15_tmp_list = []
                        val_kTau_tmp_list = []
                        val_sRho_tmp_list = []
                        for _ in range(1):
                            val_f1score, val_map50, val_map15, val_kTau, val_sRho = self.evaluate(dataloader=self.val_loader)
                            val_f1_tmp_list.append(val_f1score)
                            val_map50_tmp_list.append(val_map50)
                            val_map15_tmp_list.append(val_map15)
                            val_kTau_tmp_list.append(val_kTau)
                            val_sRho_tmp_list.append(val_sRho)
                        val_f1score = np.mean(val_f1_tmp_list)
                        val_map50 = np.mean(val_map50_tmp_list)
                        val_map15 = np.mean(val_map15_tmp_list)
                        val_kTau = np.mean(val_kTau_tmp_list)
                        val_sRho = np.mean(val_sRho_tmp_list)
                    else:
                        val_f1score, val_map50, val_map15, val_kTau, val_sRho = self.evaluate(dataloader=self.val_loader)
                        self.writer.add_scalar('f1score/val', val_f1score, epoch_i)
                        self.writer.add_scalar('map50/val', val_map50, epoch_i)
                        self.writer.add_scalar('map15/val', val_map15, epoch_i)
                        self.writer.add_scalar('kTau/val', val_kTau, epoch_i)
                        self.writer.add_scalar('sRho/val', val_sRho, epoch_i)


                    if best_f1score <= val_f1score:
                        best_f1score = val_f1score
                        best_f1score_epoch = epoch_i
                        f1_save_ckpt_path = os.path.join(self.config.save_dir_root, f"{self.config.data_type}_{self.config.combine_with_vis}", f'best_f1.pkl')
                        torch.save(self.model.state_dict(), f1_save_ckpt_path)
                        if self.config.ema == True:
                            torch.save(self.ema_model.state_dict(), f1_save_ckpt_path.split('.')[0] + '_ema.pkl')

                    if best_map50 <= val_map50:
                        best_map50 = val_map50
                        best_map50_epoch = epoch_i
                        map50_save_ckpt_path = os.path.join(self.config.save_dir_root, f"{self.config.data_type}_{self.config.combine_with_vis}", f'best_map50.pkl')
                        torch.save(self.model.state_dict(), map50_save_ckpt_path)
                        if self.config.ema == True:
                            torch.save(self.ema_model.state_dict(), map50_save_ckpt_path.split('.')[0] + '_ema.pkl')
                    
                    if best_map15 <= val_map15:
                        best_map15 = val_map15
                        best_map15_epoch = epoch_i
                        map15_save_ckpt_path = os.path.join(self.config.save_dir_root, f"{self.config.data_type}_{self.config.combine_with_vis}", f'best_map15.pkl')
                        torch.save(self.model.state_dict(), map15_save_ckpt_path)
                        if self.config.ema == True:
                            torch.save(self.ema_model.state_dict(), map15_save_ckpt_path.split('.')[0] + '_ema.pkl')
                    
                    if best_srho <= val_sRho:
                        best_srho = val_sRho
                        best_srho_epoch = epoch_i
                        srho_save_ckpt_path = os.path.join(self.config.save_dir_root, f"{self.config.data_type}_{self.config.combine_with_vis}", f'best_srho.pkl')
                        torch.save(self.model.state_dict(), srho_save_ckpt_path)
                        if self.config.ema == True:
                            torch.save(self.ema_model.state_dict(), srho_save_ckpt_path.split('.')[0] + '_ema.pkl')
                            
                    if best_ktau <= val_kTau:
                        best_ktau = val_kTau
                        best_ktau_epoch = epoch_i
                        ktau_save_ckpt_path = os.path.join(self.config.save_dir_root, f"{self.config.data_type}_{self.config.combine_with_vis}", f'best_ktau.pkl')
                        torch.save(self.model.state_dict(), ktau_save_ckpt_path)
                        if self.config.ema == True:
                            torch.save(self.ema_model.state_dict(), ktau_save_ckpt_path.split('.')[0] + '_ema.pkl')

            loss = np.mean(np.array(loss_history))
            #print(f"train epoch loss: {loss}", flush=True)
            
            if self.config.dataset == 'summe':
                val_f1_tmp_list = []
                val_map50_tmp_list = []
                val_map15_tmp_list = []
                val_kTau_tmp_list = []
                val_sRho_tmp_list = []
                for _ in range(1):
                    val_f1score, val_map50, val_map15, val_kTau, val_sRho = self.evaluate(dataloader=self.val_loader)
                    val_f1_tmp_list.append(val_f1score)
                    val_map50_tmp_list.append(val_map50)
                    val_map15_tmp_list.append(val_map15)
                    val_kTau_tmp_list.append(val_kTau)
                    val_sRho_tmp_list.append(val_sRho)
                val_f1score = np.mean(val_f1_tmp_list)
                val_map50 = np.mean(val_map50_tmp_list)
                val_map15 = np.mean(val_map15_tmp_list)
                val_kTau = np.mean(val_kTau_tmp_list)
                val_sRho = np.mean(val_sRho_tmp_list)
            else:
                val_f1score, val_map50, val_map15, val_kTau, val_sRho, val_loss = self.evaluate(dataloader=self.val_loader)
                self.writer.add_scalar('f1score/val', val_f1score, epoch_i)
                self.writer.add_scalar('map50/val', val_map50, epoch_i)
                self.writer.add_scalar('map15/val', val_map15, epoch_i)
                self.writer.add_scalar('kTau/val', val_kTau, epoch_i)
                self.writer.add_scalar('sRho/val', val_sRho, epoch_i)
                self.writer.add_scalar('loss/val', val_loss, epoch_i)
                
                if self.config.early_stop_by == 'f1score':
                    self.early_stopper(val_f1score)
                elif self.config.early_stop_by == 'map50':
                    self.early_stopper(val_map50)
                elif self.config.early_stop_by == 'map15':
                    self.early_stopper(val_map15)

                if self.early_stopper.early_stop == True:
                    break
            
            if best_f1score <= val_f1score:
                best_f1score = val_f1score
                best_f1score_epoch = epoch_i
                f1_save_ckpt_path = os.path.join(self.config.best_f1score_save_dir, f'best_f1.pkl')
                if self.config.ema == True:
                    torch.save(self.ema_model.state_dict(), f1_save_ckpt_path.split('.')[0] + '_ema.pkl')
                torch.save(self.model.state_dict(), f1_save_ckpt_path)

            if best_map50 <= val_map50:
                best_map50 = val_map50
                best_map50_epoch = epoch_i
                map50_save_ckpt_path = os.path.join(self.config.best_map50_save_dir, f'best_map50.pkl')
                if self.config.ema == True:
                    torch.save(self.ema_model.state_dict(), map50_save_ckpt_path.split('.')[0] + '_ema.pkl')
                torch.save(self.model.state_dict(), map50_save_ckpt_path)
            
            if best_map15 <= val_map15:
                best_map15 = val_map15
                best_map15_epoch = epoch_i
                map15_save_ckpt_path = os.path.join(self.config.best_map15_save_dir, f'best_map15.pkl')
                if self.config.ema == True:
                    torch.save(self.ema_model.state_dict(), map15_save_ckpt_path.split('.')[0] + '_ema.pkl')
                torch.save(self.model.state_dict(), map15_save_ckpt_path)
            
            if best_srho <= val_sRho:
                best_srho = val_sRho
                best_srho_epoch = epoch_i
                srho_save_ckpt_path = os.path.join(self.config.best_srho_save_dir, f'best_srho.pkl')
                torch.save(self.model.state_dict(), srho_save_ckpt_path)
                if self.config.ema == True:
                    torch.save(self.ema_model.state_dict(), srho_save_ckpt_path.split('.')[0] + '_ema.pkl')
                    
            if best_ktau <= val_kTau:
                best_ktau = val_kTau
                best_ktau_epoch = epoch_i
                ktau_save_ckpt_path = os.path.join(self.config.best_ktau_save_dir, f'best_ktau.pkl')
                torch.save(self.model.state_dict(), ktau_save_ckpt_path)
                if self.config.ema == True:
                    torch.save(self.ema_model.state_dict(), ktau_save_ckpt_path.split('.')[0] + '_ema.pkl')

                
            if self.scheduler != None and self.config.dataset == 'mrhisum':
                self.scheduler.step()

            self.log_gpu_memory(epoch_i)
            
            print("   [Epoch {0}] Train loss: {1:.05f}".format(epoch_i, loss))
            print('    VAL  F-score {0:0.5} | MAP50 {1:0.5} | MAP15 {2:0.5}'.format(val_f1score, val_map50, val_map15))
            print('    VAL  KTau {0:0.5} | SRho {1:0.5}'.format(val_kTau, val_sRho))
            
        print('   Best Val F1 score {0:0.5} @ epoch{1}'.format(best_f1score, best_f1score_epoch))
        print('   Best Val MAP-50   {0:0.5} @ epoch{1}'.format(best_map50, best_map50_epoch))
        print('   Best Val MAP-15   {0:0.5} @ epoch{1}'.format(best_map15, best_map15_epoch))
        print('   Best Val SRho     {0:0.5} @ epoch{1}'.format(best_srho, best_srho_epoch))
        print('   Best Val KTau     {0:0.5} @ epoch{1}'.format(best_ktau, best_ktau_epoch))

        f = open(os.path.join(self.config.save_dir_root, 'results.txt'), 'a')
        f.write(f'{self.config.data_type} - {self.config.combine_with_vis}')
        f.write('   Best Val F1 score {0:0.5} @ epoch{1}\n'.format(best_f1score, best_f1score_epoch))
        f.write('   Best Val MAP-50   {0:0.5} @ epoch{1}\n'.format(best_map50, best_map50_epoch))
        f.write('   Best Val MAP-15   {0:0.5} @ epoch{1}\n\n'.format(best_map15, best_map15_epoch))
        f.flush()
        f.close()

        return f1_save_ckpt_path, map50_save_ckpt_path, map15_save_ckpt_path, srho_save_ckpt_path, ktau_save_ckpt_path
    
    def evaluate(self, dataloader=None, test=False):
        """ Saves the frame's importance scores for the test videos in json format.

        :param int epoch_i: The current training epoch.
        """
        self.model.eval()
        
        fscore_history = []
        kTau_history = []
        sRho_history = []
        map50_history = []
        map15_history = []
        WSE_history = []
        CIS_history = []
        WIR_history = []
        IR_history = []
        loss_history = []
        
        summary_history = {}
        f1_history = {}
        

        dataloader = iter(dataloader)
        
        for data in dataloader:
            frame_features = data['features'].to(self.config.device)
            gtscore = data['gtscore'].to(self.config.device)

            if len(frame_features.shape) == 2:
                seq = seq.unsqueeze(0)
            if len(gtscore.shape) == 1:
                gtscore = gtscore.unsqueeze(0)

            B = frame_features.shape[0]
            mask=None
            if 'mask' in data:
                mask = data['mask'].to(self.config.device)
                #print(f"mask in evaluate: {mask.shape}", flush=True) # torch.Size([batch_size=1, frame_num])

            with torch.no_grad():
                if self.config.model == 'CSTA':
                    frame_features = frame_features.expand(3, -1, -1)
                    frame_features = frame_features.unsqueeze(0)
                    score = self.model(frame_features, mask).detach()
                    score = score.squeeze()[:int(data['n_frames'][0])]
                    gtscore = gtscore.squeeze()[:int(data['n_frames'][0])]
                else:
                    if self.config.model == 'A2Summ':
                        score = self.model(frame_features, mask)
                    else:
                        score, attn_weights = self.model(frame_features, mask=mask)
                    score = score.squeeze()[:int(data['n_frames'][0])]
                    gtscore = gtscore.squeeze()[:int(data['n_frames'][0])]

                loss = self.criterion(score, gtscore)
                loss = loss.mean() 
                loss_history.append(loss.item())

            
            # Summarization metric
            score = score.squeeze().cpu()
            gt_summary = data['gt_summary'][0]
            cps = data['change_points'][0]
            n_frames = data['n_frames']
            nfps = data['n_frame_per_seg'][0].tolist()
            picks = data['picks'][0]
            
            machine_summary = generate_summary(score, cps, n_frames, nfps, picks)
            summary_history[data['video_name'][0]] = machine_summary.tolist()
            if self.config.dataset == 'mrhisum':
                f_score, kTau, sRho = evaluate_summary(machine_summary, gt_summary, score, gtscore=gtscore, 
                                                    dataset=self.config.dataset,
                                                    eval_method='avg')
                
            fscore_history.append(f_score)
            kTau_history.append(kTau)
            sRho_history.append(sRho)
            f1_history[data['video_name'][0]] = f_score
            
            if test and self.config.dataset == 'mrhisum':
                WSE, CIS, WIR, IR = evaluate_knapsack_opt(score, gtscore, gt_summary, cps, n_frames, nfps, picks)
                WSE_history.append(WSE)
                CIS_history.append(CIS)
                WIR_history.append(WIR)
                IR_history.append(IR)
                            
            # Highlight Detection Metric
            gt_seg_score = generate_mrsum_seg_scores(gtscore.squeeze(0), uniform_clip=5)
            gt_top50_summary = top50_summary(gt_seg_score)
            gt_top15_summary = top15_summary(gt_seg_score)
            
            highlight_seg_machine_score = generate_mrsum_seg_scores(score, uniform_clip=5)
            highlight_seg_machine_score = torch.exp(highlight_seg_machine_score) / (torch.exp(highlight_seg_machine_score).sum() + 1e-7)
            
            clone_machine_summary = highlight_seg_machine_score.clone().detach().cpu()
            clone_machine_summary = clone_machine_summary.numpy()
            aP50 = average_precision_score(gt_top50_summary, clone_machine_summary)
            aP15 = average_precision_score(gt_top15_summary, clone_machine_summary)
            map50_history.append(aP50)
            map15_history.append(aP15)
            
        if self.config.save_results:
            with open(os.path.join(self.config.best_f1score_save_dir, 'summary_results.json'), 'w') as f:
                json.dump(summary_history, f)
            with open(os.path.join(self.config.best_f1score_save_dir, 'f1_results.json'), 'w') as f:
                json.dump(f1_history, f)
        
        final_f_score = np.mean(fscore_history)
        final_kTau = np.mean(kTau_history)
        final_sRho = np.mean(sRho_history)
        final_map50 = np.mean(map50_history)
        final_map15 = np.mean(map15_history)
        final_loss = np.mean(loss_history)

        if test and self.config.dataset == 'mrhisum':
            final_WSE = np.mean(WSE_history)
            final_CIS = np.mean(CIS_history)
            final_WIR = np.mean(WIR_history)
            final_IR = np.mean(IR_history)
            return final_f_score, final_map50, final_map15, final_kTau, final_sRho, final_WSE, final_CIS, final_WIR, final_IR, final_loss
        else:
            return final_f_score, final_map50, final_map15, final_kTau, final_sRho, final_loss

    def test(self, ckpt_path):
        if ckpt_path != None:
            print("Testing Model: ", ckpt_path)
            print("Device: ",  self.config.device)
            self.model.load_state_dict(torch.load(ckpt_path))
            if self.config.ema == True:
                self.ema_model = deepcopy(self.model).to(self.config.device)
        
        if self.config.dataset == 'mrhisum':
            test_fscore, test_map50, test_map15, test_kTau, test_sRho, test_WSE, test_CIS, test_WIR, test_IR, test_loss = self.evaluate(dataloader=self.test_loader, test=True)
            ckpt_type = ckpt_path.split('/')[-1].split('.')[0]
            self.writer.add_scalar(f'f1score/test/{ckpt_type}', test_fscore, 0)
            self.writer.add_scalar(f'map50/test/{ckpt_type}', test_map50, 0)
            self.writer.add_scalar(f'map15/test/{ckpt_type}', test_map15, 0)
            self.writer.add_scalar(f'kTau/test/{ckpt_type}', test_kTau, 0)
            self.writer.add_scalar(f'sRho/test/{ckpt_type}', test_sRho, 0)

        print("------------------------------------------------------")
        print(f"   TEST RESULT on {ckpt_path}: ")
        print('   TEST MRSum F-score {0:0.5} | MAP50 {1:0.5} | MAP15 {2:0.5}'.format(test_fscore, test_map50, test_map15))
        print('   TEST MRSum KTau {0:0.5} | SRho {1:0.5}'.format(test_kTau, test_sRho))
        if self.config.dataset == 'mrhisum':
            print('   TEST MRSum WSE {0:0.5} | CIS {1:0.5} | WIR {2:0.5} | IR {2:0.5}'.format(test_WSE, test_CIS, test_WIR, test_IR))
        print("------------------------------------------------------")
        
        f = open(os.path.join(self.config.save_dir_root, 'results.txt'), 'a')
        f.write("Testing on Model " + ckpt_path + '\n')
        f.write('Test F-score ' + str(test_fscore) + '\n')
        f.write('Test MAP50   ' + str(test_map50) + '\n')
        f.write('Test MAP15   ' + str(test_map15) + '\n\n')
        if self.config.dataset == 'mrhisum':
            f.write('Test WSE ' + str(test_WSE) + '\n')
            f.write('Test CIS ' + str(test_CIS) + '\n')
            f.write('Test WIR ' + str(test_WIR) + '\n')
            f.write('Test IR ' + str(test_IR) + '\n\n')
        f.flush()
        
        if self.config.dataset != 'mrhisum':
            return test_fscore, test_map50, test_map15, test_kTau, test_sRho
                
    @staticmethod
    def init_weights(net, init_type="xavier", init_gain=1.4142):
        """ Initialize 'net' network weights, based on the chosen 'init_type' and 'init_gain'.

        :param nn.Module net: Network to be initialized.
        :param str init_type: Name of initialization method: normal | xavier | kaiming | orthogonal.
        :param float init_gain: Scaling factor for normal.
        """
        for name, param in net.named_parameters():
            if 'weight' in name and "norm" not in name:
                if init_type == "normal":
                    nn.init.normal_(param, mean=0.0, std=init_gain)
                elif init_type == "xavier":
                    nn.init.xavier_uniform_(param, gain=np.sqrt(2.0))  # ReLU activation function
                elif init_type == "kaiming":
                    nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity="relu")
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(param, gain=np.sqrt(2.0))      # ReLU activation function
                else:
                    raise NotImplementedError(f"initialization method {init_type} is not implemented.")
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)

if __name__ == '__main__':
    pass