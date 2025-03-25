import logging
import time
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import average_precision_score

import torch
import torch.nn as nn
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter

from networks.a2summ.a2summ import *
from networks.a2summ.losses import *
from model.utils.evaluation_metrics import evaluate_summary, evaluate_knapsack_opt
from model.utils.generate_summary import generate_summary, get_gt
from model.utils.evaluate_map import generate_mrsum_seg_scores, top50_summary, top15_summary
from model.utils.early_stopping import EarlyStopping

import json


class Solver(object):
    def __init__(self, config=None, train_loader=None, val_loader=None, test_loader=None, ckpt_path=None):
        self.model, self.optimizer, self.writer, self.scheduler = None, None, None, None
    
        self.config = config
        self.config.weight_decay = float(self.config.weight_decay)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.global_step = 0
        self.ckpt_path = ckpt_path

    def build(self):
        if self.config.model == 'A2Summ':
            self.model = Model_BLiSS(config=self.config)
            self.model.to(self.config.device)
            self.calc_contrastive_loss = Dual_Contrastive_Loss().to(self.config.device)

            self.parameters = [p for p in self.model.parameters() if p.requires_grad] + \
                        [p for p in self.calc_contrastive_loss.parameters() if p.requires_grad]

            self.optimizer = torch.optim.Adam(self.parameters, lr=self.config.lr, weight_decay=self.config.weight_decay)

        else:
            print("Wrong model")
            exit()

        self.early_stopper = EarlyStopping()
        self.write = SummaryWriter(log_dir = f'{self.config.tensorboard_path}/{self.config.tag}')

        if self.ckpt_path != None:
            print("Loading Model: ", self.ckpt_path)
            self.model.load_state_dict(torch.load(self.ckpt_path))
            self.model.to(self.config.device)


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


        for epoch_i in range(self.config.epochs):
            print("[Epoch: {0:6}]".format(str(epoch_i)+"/"+str(self.config.epochs)))
            self.model.train()

            loss_history = []
            text_loss_history = []
            vis_loss_history = []
            inter_contrastive_loss_history = []
            intra_contrastive_loss_history = []

            num_batches = int(len(self.train_loader))
            iterator = iter(self.train_loader)

            for batch_idx in tqdm(range(num_batches)):
                self.optimizer.zero_grad()
                data = next(iterator)

                vis_feature = data['vis_feature'].to(self.config.device)
                if self.config.mix_type == 'vt':
                    other_feature = data['text_feature'].to(self.config.device)
                    other_feature_name = 'text'
                elif self.config.mix_type == 'va':
                    other_feature = data['audio_feature'].to(self.config.device)
                    other_feature_name = 'audio'
                else:
                    raise ValueError("A2summ accepts only 'vt' or 'va' as mix_type")


                batch_size = len(vis_feature)

                for i in range(batch_size): #for each video
                    data['vis_to_text_mask'][i] = data['vis_to_text_mask'][i].to(self.config.device)
                    data['text_to_vis_mask'][i] = data['text_to_vis_mask'][i].to(self.config.device)

                data['mask'] = data['mask'].to(self.config.device)
                data['vis_label'], data['text_label'] = data['vis_label'].to(self.config.device), data['text_label'].to(self.config.device)

                pred_vis, pred_text, contrastive_pairs = self.model(video=vis_feature, text=other_feature, \
                                                                    mask_video=data['mask'], mask_text=data['mask'], \
                                                                    video_label=data['vis_label'], text_label=data['text_label'], \
                                                                    video_to_text_mask_list=data['vis_to_text_mask'], \
                                                                    text_to_video_mask_list=data['text_to_vis_mask'])

                num_frame_selected = torch.sum(data['vis_label'], dim=-1)
                num_sentence_selected = torch.sum(data['text_label'], dim=-1)

                mask_bool = data['mask'].to(torch.bool)

                # select frames and sentences with top-k highest importance score as predicted video and text summary
                keyframe_index_list = []
                keysentence_index_list = []
                for i in range(batch_size):
                    keyframe_index_list.append(torch.topk(pred_vis[i, mask_bool[i]], k=num_frame_selected[i])[1].tolist())
                    keysentence_index_list.append(torch.topk(pred_text[i, mask_bool[i]], k=num_sentence_selected[i])[1].tolist())

                text_loss = calc_cls_loss(pred_text, data['text_label'], mask=data['mask'])
                vis_loss = calc_cls_loss(pred_vis, data['vis_label'], mask=data['mask'])

                inter_contrastive_loss, intra_contrastive_loss = self.calc_contrastive_loss(contrastive_pairs)
                
                inter_contrastive_loss = inter_contrastive_loss * self.config.lambda_contrastive_inter
                intra_contrastive_loss = intra_contrastive_loss * self.config.lambda_contrastive_intra
                loss = vis_loss + text_loss + inter_contrastive_loss + intra_contrastive_loss

                loss_history.append(loss.item())
                vis_loss_history.append(vis_loss.item())
                text_loss_history.append(text_loss.item())
                inter_contrastive_loss_history.append(inter_contrastive_loss.item())
                intra_contrastive_loss_history.append(intra_contrastive_loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss = np.mean(np.array(loss_history))
            epoch_vis_loss = np.mean(np.array(vis_loss_history))
            epoch_text_loss = np.mean(np.array(text_loss_history))
            epoch_inter_contrastive_loss = np.mean(np.array(inter_contrastive_loss_history))
            epoch_intra_contrastive_loss = np.mean(np.array(intra_contrastive_loss_history))
                                
            print(f'[Train] Epoch: {epoch_i+1}/{self.config.max_epoch} LR: {self.config.lr:.4f} '
                  f'Loss: {epoch_loss:.4f}/ vis loss: {epoch_vis_loss:.4f}/ {other_feature_name} loss: {epoch_text_loss:.4f}/ inter contrastive loss: {epoch_inter_contrastive_loss:.4f}/ intra contrastive loss: {epoch_intra_contrastive_loss:.4f}', flush=True)
            
            self.writer.add_scalar('loss/train', epoch_loss, epoch_i)
            self.writer.add_scalar('vis_loss/train', epoch_vis_loss, epoch_i)
            self.writer.add_scalar(f'{other_feature_name}_loss/train', epoch_text_loss, epoch_i)
            self.writer.add_scalar('inter_contrastive_loss/train', epoch_inter_contrastive_loss, epoch_i)
            self.writer.add_scalar('intra_contrastive_loss/train', epoch_intra_contrastive_loss, epoch_i)

            val_f1score, val_map50, val_map15, val_kTau, val_sRho, val_loss = self.evaluate_msmo(self.model, self.val_loader, self.config, epoch=epoch_i, test=False)
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
                torch.save(self.model.state_dict(), f1_save_ckpt_path)

            if best_map50 <= val_map50:
                best_map50 = val_map50
                best_map50_epoch = epoch_i
                map50_save_ckpt_path = os.path.join(self.config.best_map50_save_dir, f'best_map50.pkl')
                torch.save(self.model.state_dict(), map50_save_ckpt_path)
            
            if best_map15 <= val_map15:
                best_map15 = val_map15
                best_map15_epoch = epoch_i
                map15_save_ckpt_path = os.path.join(self.config.best_map15_save_dir, f'best_map15.pkl')
                torch.save(self.model.state_dict(), map15_save_ckpt_path)
            
            if best_srho <= val_sRho:
                best_srho = val_sRho
                best_srho_epoch = epoch_i
                srho_save_ckpt_path = os.path.join(self.config.best_srho_save_dir, f'best_srho.pkl')
                torch.save(self.model.state_dict(), srho_save_ckpt_path)
                    
            if best_ktau <= val_kTau:
                best_ktau = val_kTau
                best_ktau_epoch = epoch_i
                ktau_save_ckpt_path = os.path.join(self.config.best_ktau_save_dir, f'best_ktau.pkl')
                torch.save(self.model.state_dict(), ktau_save_ckpt_path)

            if self.scheduler != None:
                self.scheduler.step()
    
            self.log_gpu_memory(epoch_i)
            self.log_cpu_ram_usage(epoch_i)

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


    @torch.no_grad()
    def evaluate_msmo(self, model, val_loader, args, epoch=None, test=False):
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
        
        dataloader = iter(val_loader)
        
        for data in dataloader:
            vis_feature = data['vis_feature'].to(self.config.device)
            if self.config.mix_type == 'vt':
                other_feature = data['text_feature'].to(self.config.device)
                other_feature_name = 'text'
            elif self.config.mix_type == 'va':
                other_feature = data['audio_feature'].to(self.config.device)
                other_feature_name = 'audio'
            else:
                raise ValueError("A2summ accepts only 'vt' or 'va' as mix_type")
            
            for i in range(len(data['vis_to_text_mask'])):
                data['vis_to_text_mask'][i] = data['vis_to_text_mask'][i].to(args.device)
                data['text_to_vis_mask'][i] = data['text_to_vis_mask'][i].to(args.device)

            data['mask'] = data['mask'].to(self.config.device)
            data['vis_label'], data['text_label'] = data['vis_label'].to(self.config.device), data['text_label'].to(self.config.device)

            pred_vis, pred_text, contrastive_pairs = model(video=vis_feature, text=other_feature, \
                                                                mask_video=data['mask'], mask_text=data['mask'], \
                                                                video_label=data['vis_label'], text_label=data['text_label'], \
                                                                video_to_text_mask_list=data['vis_to_text_mask'], \
                                                                text_to_video_mask_list=data['text_to_vis_mask'])

            num_frame_selected = torch.sum(data['vis_label'], dim=-1)
            num_sentence_selected = torch.sum(data['text_label'], dim=-1)

            mask_bool = data['mask'].to(torch.bool)

            #validation loss
            text_loss = calc_cls_loss(pred_text, data['text_label'], mask=data['mask'])
            vis_loss = calc_cls_loss(pred_vis, data['vis_label'], mask=data['mask'])
            inter_contrastive_loss, intra_contrastive_loss = self.calc_contrastive_loss(contrastive_pairs)
            
            inter_contrastive_loss = inter_contrastive_loss * self.config.lambda_contrastive_inter
            intra_contrastive_loss = intra_contrastive_loss * self.config.lambda_contrastive_intra
            loss = vis_loss + text_loss + inter_contrastive_loss + intra_contrastive_loss
            loss_history.append(loss.item())

            #summarization metric
            pred_vis = pred_vis.squeeze().cpu()
            
            machine_summary = generate_summary(pred_vis, data['change_points'], data['n_frames'], data['n_frame_per_seg'], data['picks'])
            summary_history[data['video_id'][0]] = machine_summary.tolist()

            f_score, kTau, sRho = evaluate_summary(machine_summary, data['gt_summary'], pred_vis, gtscore=data['gtscore'],
                                                   dataset='mrhisum',
                                                   eval_method='avg')
            
            fscore_history.append(f_score)
            kTau_history.append(kTau)
            sRho_history.append(sRho)
            f1_history[data['video_id'][0]] = f_score

            if test:
                WSE, CIS, WIR, IR = evaluate_knapsack_opt(pred_vis, data['gtscore'], data['gt_summary'], data['change_points'], data['n_frames'], data['n_frame_per_seg'], data['picks'])
                WSE_history.append(WSE)
                CIS_history.append(CIS)
                WIR_history.append(WIR)
                IR_history.append(IR)

            # Highlight Detection Metric
            gt_seg_score = generate_mrsum_seg_scores(data['gtscore'].squeeze(0), uniform_clip=5)
            gt_top50_summary = top50_summary(gt_seg_score)
            gt_top15_summary = top15_summary(gt_seg_score)
            
            highlight_seg_machine_score = generate_mrsum_seg_scores(pred_vis, uniform_clip=5)
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

        if test:
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

        test_fscore, test_map50, test_map15, test_kTau, test_sRho, test_WSE, test_CIS, test_WIR, test_IR, test_loss = self.evaluate_msmo(self.model, self.test_loader, self.config, test=True)
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
        print('   TEST MRSum WSE {0:0.5} | CIS {1:0.5} | WIR {2:0.5} | IR {2:0.5}'.format(test_WSE, test_CIS, test_WIR, test_IR))
        print("------------------------------------------------------")


        f = open(os.path.join(self.config.save_dir_root, 'results.txt'), 'a')
        f.write("Testing on Model " + ckpt_path + '\n')
        f.write('Test F-score ' + str(test_fscore) + '\n')
        f.write('Test MAP50   ' + str(test_map50) + '\n')
        f.write('Test MAP15   ' + str(test_map15) + '\n\n')
        f.write('Test WSE ' + str(test_WSE) + '\n')
        f.write('Test CIS ' + str(test_CIS) + '\n')
        f.write('Test WIR ' + str(test_WIR) + '\n')
        f.write('Test IR ' + str(test_IR) + '\n\n')
        f.flush()


if __name__ == '__main__':
    pass