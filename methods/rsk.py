import math
import logging
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from tqdm import tqdm
from copy import deepcopy
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from models.net_rsk import Net
from models.vit_rsk import VisionTransformer, PatchEmbed, Block, Attention_LoRA
from methods.base import BaseLearner
from utils.toolkit import tensor2numpy
from utils.toolkit import print_trainable_params, check_params_consistency


class SplitLoRAV6(BaseLearner):

    def __init__(self, args):
        super().__init__(args)

        self.topk = 1
        self.network = Net(args)

        # inflora
        self.lamb = args["lamb"]
        self.lame = args["lame"]
        self.all_keys = []
        self.feature_list = []
        self.project_type = []
        self.in_dim = args['in_dim']
        self.alpha = args['alpha']
        self._protos = []
        self.rank = args['rank']

        self.ema_type = args.get('ema_type', "FALSE")
        self.R = args.get('R', 0.5) 

        self.ortho_loss_typev2 = args.get('ortho_loss_typev2', False)
        self.lambda_o = args.get("lambda_o", 0)  
        self.mu = args.get("mu", 0.1)               
     
        

    def incremental_train(self, data_manager):
        self.data_manager = data_manager  
        self.build_train_loader(data_manager)
        logging.info('Task {} learning on class {}-{}'.format(self.cur_task, self.known_classes, self.total_classes))
        self._train(self.train_loader)

    def _extract_vectors(self, loader):
        self.network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self.network, nn.DataParallel):
                _vectors = tensor2numpy(self.network.module.extract_vector(_inputs.to(self.device)))
            else:
                _vectors = tensor2numpy(self.network.extract_vector(_inputs.to(self.device)))

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

   
    

                
    def _train(self, train_loader):
        self.network.to(self.device)
        self.freeze_network()
        print_trainable_params(self.network)

        # # Design LoRA matrix through Equation (8)
        # with torch.no_grad():
        #     self.init_drm(train_loader)

        if len(self.multiple_gpus) > 1:
            self.network = nn.DataParallel(self.network, self.multiple_gpus)

        optimizer, scheduler = self.build_optimizer(self.network.parameters())
        check_params_consistency(self.network, optimizer)



        self._train_function(train_loader, optimizer, scheduler)
        
        if len(self.multiple_gpus) > 1:
            self.network = self.network.module



    def _train_function(self, train_loader, optimizer, scheduler):
        ema_keys = ["Lora_shared_B_k", "Lora_shared_B_v", "Lora_shared_A_k", "Lora_shared_A_v"]

        if self.ema_type == "EMA" and self.cur_task>=1:
            import math

            ema_decay = math.pow(self.R, 1/ (len(train_loader)*10))

            print(f"curent task {self.cur_task}, ema decay {ema_decay}")
            self.ema_params = {}
            for name, param in self.network.named_parameters():
                if any(key in name for key in ema_keys):
                    self.ema_params[name] = param.data.clone().detach()
                    print(f"EMA tracking: {name}")


        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self.network.train()
            losses = 0.
            correct, total = 0, 0

            v2_nuc_sum = 0.0
            v2_l1_sum = 0.0
            v2_fid_sum = 0.0
            v2_batches = 0


            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                mask = (targets >= self.known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                # labels = torch.index_select(targets, 0, mask)

                targets = torch.index_select(targets, 0, mask)-self.known_classes
                ret = self.network(inputs)
                logits = ret['logits']
                loss = F.cross_entropy(logits, targets)

                if self.ortho_loss_typev2 and self.cur_task >= 1:

                    nuclear_loss = 0.0
                    # l1_loss = 0.0
                    fidelity_loss = 0.0

                    shared_count = 0
                    task_count = 0


                    for name, param in self.network.named_parameters():

                        if name in self.ema_params:

                            # Current shared adapter
                            Z_o = param

                            # Low-rank regularization
                            nuclear_loss += torch.norm(Z_o, p='fro')

                            # Historical shared knowledge
                            Z_o_ema = self.ema_params[name].detach()

                            # Fidelity / stability constraint
                            fidelity_loss += torch.norm(
                                Z_o - Z_o_ema,
                                p='fro'
                            ) ** 2
                            shared_count += 1



                    # ============================================================
                    # 3. Normalize by parameter groups
                    # ============================================================
                    if shared_count > 0:
                        nuclear_loss = nuclear_loss / shared_count
                        fidelity_loss = fidelity_loss / shared_count

                    # if task_count > 0:
                    #     l1_loss = l1_loss / task_count

                    # ============================================================
                    # 4. Unified objective
                    # ============================================================
                    loss += (
                        self.lambda_o * nuclear_loss
                        # + self.lambda_s * l1_loss
                        + (self.mu) * fidelity_loss
                    )

                    v2_nuc_sum += self.lambda_o * nuclear_loss
                    # v2_l1_sum += self.lambda_s * l1_loss
                    v2_fid_sum += self.mu * 0.5* fidelity_loss
                    v2_batches += 1
                


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.ema_type == "EMA" and self.cur_task>=1:
                # if self.cur_task>=1:
                    for name, param in self.network.named_parameters():
                        if name in self.ema_params:
                            self.ema_params[name] = ema_decay * self.ema_params[name] + (1 - ema_decay) * param.data
             
              
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if v2_batches > 0:
                print('[v2loss] Task {}, Epoch {}/{} => Loss {:.3f} | lo*nuc {:.4f}  ls*l1 {:.4f}  mu*fid {:.4f}'.format(
                    self.cur_task, epoch + 1, self.epochs, losses / len(train_loader),
                    v2_nuc_sum / v2_batches, v2_l1_sum / v2_batches, v2_fid_sum / v2_batches))

            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self.cur_task, epoch + 1, self.epochs, losses / len(train_loader), train_acc)
            prog_bar.set_description(info)

        logging.info(info)
        if self.ema_type == "EMA" and self.cur_task>=1:
            with torch.no_grad():
                for name, param in self.network.named_parameters():
                    if name in self.ema_params:
                        param.data.copy_(self.ema_params[name])
                        print(f"Applied EMA to {name}")

    def freeze_network(self):
        target_suffix = f".{self.cur_task}"
        if self.cur_task==0:
            unfrozen_keys = [
            f"classifier_pool{target_suffix}",
            "Lora_shared",
        ]
        else:
            unfrozen_keys = [
            f"classifier_pool{target_suffix}",
            "Lora_shared",
        ]
        for name, param in self.network.named_parameters():
            param.requires_grad_(any(key in name for key in unfrozen_keys))
    
    

    def _compute_accuracy_domain(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = model(inputs)['logits']

            predicts = torch.max(outputs, dim=1)[1]
            correct += ((predicts % self.class_num).cpu() == (targets % self.class_num)).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)




