import logging
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import FinetuneIncrementalNet
from torchvision import transforms
from torch.distributions.multivariate_normal import MultivariateNormal
import random
from utils.toolkit import tensor2numpy, accuracy
import copy
import os
import pdb

epochs = 20
lrate = 0.01 
milestones = [60,100,140]
lrate_decay = 0.1
# batch_size = 128
batch_size = 64
split_ratio = 0.1
T = 2
weight_decay = 5e-4
num_workers = 8


class SLCA_ori(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = FinetuneIncrementalNet(args['convnet_type'], pretrained=True)
        self.log_path = "logs/{}_{}".format(args['model_name'], args['model_postfix'])
        self.model_prefix = args['prefix']
        if 'epochs' in args.keys():
            global epochs
            epochs = args['epochs'] 
        if 'milestones' in args.keys():
            global milestones
            milestones = args['milestones']
        if 'lr' in args.keys():
            global lrate
            lrate = args['lr']
            print('set lr to ', lrate)
        if 'bcb_lrscale' in args.keys():
            self.bcb_lrscale = args['bcb_lrscale']
        else:
            self.bcb_lrscale = 1.0/100
        if self.bcb_lrscale == 0:
            self.fix_bcb = True
        else:
            self.fix_bcb = False
        print('fic_bcb', self.fix_bcb)


        
        if 'save_before_ca' in args.keys() and args['save_before_ca']:
            self.save_before_ca = True
        else:
            self.save_before_ca = False

        if 'ca_epochs' in args.keys():
            global ca_epochs
            ca_epochs = args['ca_epochs'] 

        if 'ca_with_logit_norm' in args.keys() and args['ca_with_logit_norm']>0:
            self.logit_norm = args['ca_with_logit_norm']
        else:
            self.logit_norm = None

        self.run_id = args['run_id']
        self.seed = args['seed']
        self.task_sizes = []

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))
        self.save_checkpoint(self.log_path+'/'+self.model_prefix+'_seed{}'.format(self.seed), head_only=self.fix_bcb)
        self._network.fc.recall()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        task_size = data_manager.get_task_size(self._cur_task)
        self.task_sizes.append(task_size)
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.topk = self._total_classes if self._total_classes<5 else 5
        self._network.update_fc(data_manager.get_task_size(self._cur_task))
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        self._network.to(self._device)

        train_dset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                  source='train', mode='train',
                                                  appendent=[], with_raw=False)
        test_dset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        dset_name = data_manager.dataset_name.lower()

        self.train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        self._stage1_training(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        # CA
        self._network.fc.backup()
        if self.save_before_ca:
            self.save_checkpoint(self.log_path+'/'+self.model_prefix+'_seed{}_before_ca'.format(self.seed), head_only=self.fix_bcb)
        
        self._compute_class_mean(data_manager, check_diff=False, oracle=False)
        if self._cur_task>0 and ca_epochs>0:
            self._stage2_compact_classifier(task_size)
            if len(self._multiple_gpus) > 1:
                self._network = self._network.module
        

    def _run(self, train_loader, test_loader, optimizer, scheduler):
        run_epochs = epochs
        for epoch in range(1, run_epochs+1):
            self._network.train()
            losses = 0.
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                logits = self._network(inputs, bcb_no_grad=self.fix_bcb)['logits']
                cur_targets = torch.where(targets-self._known_classes>=0,targets-self._known_classes,-100)
                loss = F.cross_entropy(logits[:, self._known_classes:], cur_targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            if epoch%5==0:
                train_acc = self._compute_accuracy(self._network, train_loader)
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.3f}, Test_accy {:.3f}'.format(
                    self._cur_task, epoch, epochs, losses/len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}'.format(
                    self._cur_task, epoch, epochs, losses/len(train_loader))
            logging.info(info)

    def _stage1_training(self, train_loader, test_loader):
        '''
        if self._cur_task == 0:
            loaded_dict = torch.load('./dict_0.pkl')
            self._network.load_state_dict(loaded_dict['model_state_dict'])
            self._network.to(self._device)
            return
        '''
        base_params = self._network.convnet.parameters()
        base_fc_params = [p for p in self._network.fc.parameters() if p.requires_grad==True]
        head_scale = 1. if 'moco' in self.log_path else 1.
        if not self.fix_bcb:
            base_params = {'params': base_params, 'lr': lrate*self.bcb_lrscale, 'weight_decay': weight_decay}
            base_fc_params = {'params': base_fc_params, 'lr': lrate*head_scale, 'weight_decay': weight_decay}
            network_params = [base_params, base_fc_params]
        else:
            for p in base_params:
                p.requires_grad = False
            network_params = [{'params': base_fc_params, 'lr': lrate*head_scale, 'weight_decay': weight_decay}]
        optimizer = optim.SGD(network_params, lr=lrate, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._run(train_loader, test_loader, optimizer, scheduler)


    def _stage2_compact_classifier(self, task_size):
        for p in self._network.fc.parameters():
            p.requires_grad=True
            
        run_epochs = ca_epochs
        crct_num = self._total_classes    
        param_list = [p for p in self._network.fc.parameters() if p.requires_grad]
        network_params = [{'params': param_list, 'lr': lrate,
                           'weight_decay': weight_decay}]
        optimizer = optim.SGD(network_params, lr=lrate, momentum=0.9, weight_decay=weight_decay)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[4], gamma=lrate_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)

        self._network.to(self._device)
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._network.eval()
        for epoch in range(run_epochs):
            losses = 0.

            sampled_data = []
            sampled_label = []
            num_sampled_pcls = 256
        
            for c_id in range(crct_num):
                t_id = c_id//task_size
                decay = (t_id+1)/(self._cur_task+1)*0.1
                cls_mean = torch.tensor(self._class_means[c_id], dtype=torch.float64).to(self._device)*(0.9+decay) # torch.from_numpy(self._class_means[c_id]).to(self._device)
                cls_cov = self._class_covs[c_id].to(self._device)
                
                m = MultivariateNormal(cls_mean.float(), cls_cov.float())

                sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                sampled_data.append(sampled_data_single)                
                sampled_label.extend([c_id]*num_sampled_pcls)

            sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device)
            sampled_label = torch.tensor(sampled_label).long().to(self._device)

            inputs = sampled_data
            targets= sampled_label

            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            
            for _iter in range(crct_num):
                inp = inputs[_iter*num_sampled_pcls:(_iter+1)*num_sampled_pcls]
                tgt = targets[_iter*num_sampled_pcls:(_iter+1)*num_sampled_pcls]
                outputs = self._network(inp, bcb_no_grad=True, fc_only=True)
                logits = outputs['logits']

                if self.logit_norm is not None:
                    per_task_norm = []
                    prev_t_size = 0
                    cur_t_size = 0
                    for _ti in range(self._cur_task+1):
                        cur_t_size += self.task_sizes[_ti]
                        temp_norm = torch.norm(logits[:, prev_t_size:cur_t_size], p=2, dim=-1, keepdim=True) + 1e-7
                        per_task_norm.append(temp_norm)
                        prev_t_size += self.task_sizes[_ti]
                    per_task_norm = torch.cat(per_task_norm, dim=-1)
                    norms = per_task_norm.mean(dim=-1, keepdim=True)
                        
                    norms_all = torch.norm(logits[:, :crct_num], p=2, dim=-1, keepdim=True) + 1e-7
                    decoupled_logits = torch.div(logits[:, :crct_num], norms) / self.logit_norm
                    loss = F.cross_entropy(decoupled_logits, tgt)

                else:
                    loss = F.cross_entropy(logits[:, :crct_num], tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            test_acc = self._compute_accuracy(self._network, self.test_loader)
            info = 'CA Task {} => Loss {:.3f}, Test_accy {:.3f}'.format(
                self._cur_task, losses/self._total_classes, test_acc)
            logging.info(info)
# {
#     "prefix": "reproduce",
#     "dataset": "cifar100_224",
#     "memory_size": 0,
#     "memory_per_class": 0,
#     "fixed_memory": false,
#     "shuffle": true,
#     "init_cls": 10,
#     "increment": 10,
#     "model_name": "slca_cifar",
#     "model_postfix": "20e",
#     "convnet_type": "vit-b-p16",
#     "device": ["2"],
#     "seed": [1993],
#     "epochs": 20,
#     "ca_epochs": 5,
#     "ca_with_logit_norm": 0.1,
#     "milestones": [18]
# }

# log_cifar100_ourbase_onlytrain_qkvmlp
class SLCA_old(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = FinetuneIncrementalNet(args['convnet_type'], pretrained=True)
        self.log_path = "logs/{}_{}".format(args['model_name'], args['model_postfix'])
        self.model_prefix = args['prefix']
        if 'epochs' in args.keys():
            global epochs
            epochs = args['epochs'] 
        if 'milestones' in args.keys():
            global milestones
            milestones = args['milestones']
        if 'lr' in args.keys():
            global lrate
            lrate = args['lr']
            print('set lr to ', lrate)
        if 'bcb_lrscale' in args.keys():
            self.bcb_lrscale = args['bcb_lrscale']
        else:
            self.bcb_lrscale = 1.0/100
        self.fix_bcb = False
        print('fic_bcb', self.fix_bcb)


        if 'save_before_ca' in args.keys() and args['save_before_ca']:
            self.save_before_ca = True
        else:
            self.save_before_ca = False

        if 'ca_epochs' in args.keys():
            global ca_epochs
            ca_epochs = args['ca_epochs'] 

        if 'ca_with_logit_norm' in args.keys() and args['ca_with_logit_norm']>0:
            self.logit_norm = args['ca_with_logit_norm']
        else:
            self.logit_norm = None

        self.run_id = args['run_id']
        self.seed = args['seed']
        self.task_sizes = []
        
        self.cnn_acc, self.nme_acc = [], []

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))
        # self.save_checkpoint(self.log_path+'/'+self.model_prefix+'_seed{}'.format(self.seed), head_only=self.fix_bcb)
        self._network.fc.recall()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        task_size = data_manager.get_task_size(self._cur_task)
        self.task_sizes.append(task_size)
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.topk = self._total_classes if self._total_classes<5 else 5
        self._network.update_fc(data_manager.get_task_size(self._cur_task))
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        self._network.to(self._device)

        train_dset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                  source='train', mode='train',
                                                  appendent=[], with_raw=False)
        test_dset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        dset_name = data_manager.dataset_name.lower()

        self.train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        self._stage1_training(self.train_loader, self.test_loader)
        
        self.build_rehearsal_memory(data_manager, per_class=30)
        buffer_dataset = data_manager.get_dataset([], source='train', mode='test', appendent=self._get_memory())
        self.buffer_loader = DataLoader(buffer_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self._class_means = self.calculate_class_means(self._network, self.buffer_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        # CA
        self._network.fc.backup()
        # if self.save_before_ca:
        #     self.save_checkpoint(self.log_path+'/'+self.model_prefix+'_seed{}_before_ca'.format(self.seed), head_only=self.fix_bcb)
        
        # self._compute_class_mean(data_manager, check_diff=False, oracle=False)
        # if self._cur_task>0 and ca_epochs>0:
        #     # self._stage2_compact_classifier(task_size)
        #     if len(self._multiple_gpus) > 1:
        #         self._network = self._network.module
        
    def _run(self, train_loader, test_loader, optimizer, scheduler):
        run_epochs = epochs
        for epoch in range(1, run_epochs+1):
            self._network.train()
            losses = 0.
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                # pdb.set_trace()
                logits = self._network(inputs, bcb_no_grad=self.fix_bcb)['logits']
                cur_targets = torch.where(targets-self._known_classes>=0,targets-self._known_classes,-100)
                loss = F.cross_entropy(logits[:, self._known_classes:], cur_targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            if epoch%5==0:
                train_acc = self._compute_accuracy(self._network, train_loader)
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.3f}, Test_accy {:.3f}'.format(
                    self._cur_task, epoch, epochs, losses/len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}'.format(
                    self._cur_task, epoch, epochs, losses/len(train_loader))
            logging.info(info)

    def _stage1_training(self, train_loader, test_loader):
        '''
        if self._cur_task == 0:
            loaded_dict = torch.load('./dict_0.pkl')
            self._network.load_state_dict(loaded_dict['model_state_dict'])
            self._network.to(self._device)
            return
        '''
        
        enable = []
        for name, param in self._network.named_parameters():
            # only train FC
            param.requires_grad_(False)
            # if ("mlp" in name)or("qkv" in name)or("attn.proj" in name):
            if ("mlp" in name)or("qkv" in name)or("fc.heads" in name):
                param.requires_grad_(True)
                enable.append(name)
        if self._cur_task == 0:
            print(enable)
        # pdb.set_trace()
        
        base_params = self._network.convnet.parameters()
        base_fc_params = [p for p in self._network.fc.parameters() if p.requires_grad==True]
        head_scale = 1. if 'moco' in self.log_path else 1.
        if not self.fix_bcb:
            # base_params = []
            # for name, param in self._network.named_parameters():
            #     if ("mlp" in name)or("qkv" in name):
            #         base_params.append(param)
            # base_params = {'params': base_params, 'lr': 0.001, 'weight_decay': weight_decay}
            # base_fc_params = {'params': base_fc_params, 'lr': 0.001, 'weight_decay': weight_decay}
            
            base_params = {'params': base_params, 'lr': lrate*self.bcb_lrscale, 'weight_decay': weight_decay}
            base_fc_params = {'params': base_fc_params, 'lr': lrate*head_scale, 'weight_decay': weight_decay}
            network_params = [base_params, base_fc_params]
        else:
            for p in base_params:
                p.requires_grad = False
            network_params = [{'params': base_fc_params, 'lr': lrate*head_scale, 'weight_decay': weight_decay}]
        optimizer = optim.SGD(network_params, lr=lrate, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        
        self._run(train_loader, test_loader, optimizer, scheduler)

    def calculate_class_means(self, model, data_loader):
        # calculate class means in train loader
        # class_means = torch.zeros(self._total_classes, self._network.feature_dim)
        print('Calculating class means...')
        model.eval()
        total = [0]*self._total_classes
        all_features = [[] for i in range(self._total_classes)]
        for i, (_, inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            with torch.no_grad():
                image_features = model(inputs)['features']
            vectors = image_features.clone().detach().cpu().numpy()
            for j in range(len(targets)):
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + 1e-8)).T
                total[targets[j]] += 1
                all_features[targets[j]].append(torch.tensor(vectors[j]))
        class_means = []
        for i in range(self._total_classes):
            features = torch.stack(all_features[i], dim=0).sum(dim=0)
            class_means.append(features / total[i])
        class_means = F.normalize(torch.stack(class_means, dim=0), p=2, dim=-1)
        return class_means


def _KD_loss(pred, soft, T, weight=None):
    if weight == None:
        weight = torch.ones(len(pred)).to(pred.device)
    pred = torch.log_softmax(pred/T, dim=1)
    soft = torch.softmax(soft/T, dim=1)
    return torch.tensor(0)
    # return -1 * (torch.mul(soft, pred)*weight.unsqueeze(1)).sum()/pred.shape[0]

from PIL import Image
from utils.data_manager import pil_loader
from torch.utils.data import Dataset
class DummyDataset1(Dataset):
    def __init__(self, images, labels, idx, trsf, use_path=False):
        assert len(images) == len(labels), 'Data size error!'
        self.images = images
        self.labels = labels
        self.idx = idx
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return self.idx[idx], image, label


# our,先不在训练中加入保存的样本，只在slca基础上加上稀疏
# log_cifar100_ourmethods_no_replay
class SLCA_no_replay(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = FinetuneIncrementalNet(args['convnet_type'], pretrained=True)
        self.log_path = "logs/{}_{}".format(args['model_name'], args['model_postfix'])
        self.model_prefix = args['prefix']
        if 'epochs' in args.keys():
            global epochs
            epochs = args['epochs'] 
        if 'milestones' in args.keys():
            global milestones
            milestones = args['milestones']
        if 'lr' in args.keys():
            global lrate
            lrate = args['lr']
            print('set lr to ', lrate)
        if 'bcb_lrscale' in args.keys():
            self.bcb_lrscale = args['bcb_lrscale']
        else:
            self.bcb_lrscale = 1.0/100
        self.fix_bcb = False
        print('fic_bcb', self.fix_bcb)


        if 'save_before_ca' in args.keys() and args['save_before_ca']:
            self.save_before_ca = True
        else:
            self.save_before_ca = False

        if 'ca_epochs' in args.keys():
            global ca_epochs
            ca_epochs = args['ca_epochs'] 

        if 'ca_with_logit_norm' in args.keys() and args['ca_with_logit_norm']>0:
            self.logit_norm = args['ca_with_logit_norm']
        else:
            self.logit_norm = None

        self.run_id = args['run_id']
        self.seed = args['seed']
        self.task_sizes = []
        
        self.args = args
        self.model = 'vit'
        self.cnn_acc, self.nme_acc = [], []
        self.remain_params = []
        self.args['update_step'] = 5
        self.gamma = 1
        self.args["use_faster_sample_grad"] = True
        self.args["use_param_grad"] = True

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))
        # self.save_checkpoint(self.log_path+'/'+self.model_prefix+'_seed{}'.format(self.seed), head_only=self.fix_bcb)
        self._network.fc.recall()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        task_size = data_manager.get_task_size(self._cur_task)
        self.task_sizes.append(task_size)
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.topk = self._total_classes if self._total_classes<5 else 5
        self._network.update_fc(data_manager.get_task_size(self._cur_task))
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))
        
        self.mask_ratio_per_task = 0.03
        self.mask_ratio = 1-self.mask_ratio_per_task*(self._cur_task+1)
        enabled, mask_enabled = set(), set()
        if self.model == 'vit':
            # qkv+mlp+proj:84934656 qkv+mlp: 77856768; qkv: 21233664
            for name, param in self._network.named_parameters():
                param.requires_grad_(False)
                if ("fc.head" in name):
                    param.requires_grad_(True)
                # if ("mlp" in name)or("qkv" in name)or("attn.proj" in name):
                if ("mlp" in name)or("qkv" in name):
                    param.requires_grad_(True)
                if param.requires_grad:
                    enabled.add(name)
            for name, m in self._network.named_modules():
                if ("mlp.fc" in name)or("qkv" in name):  
                # if hasattr(m, "weight_mask"):
                    m.set_mask_ratio(self.mask_ratio)
                    mask_enabled.add(name)
        if self._cur_task == 0:
            logging.info('Parameters to be updated: {}'.format(enabled))
            logging.info('Parameters to be masked: {}'.format(mask_enabled))
        if self.args['dataset'] in ['core50', 'domainnet']:
            self._total_classes_test = 50 if self.args['dataset']=='core50' else 200
        else:
            self._total_classes_test = self._total_classes

        data, targets, train_dset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                  source='train', mode='train',
                                                  appendent=[], ret_data=True)
        test_dset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        dset_name = data_manager.dataset_name.lower()
        
        self._network.reset_data_mask(len(data))
        idx = np.array([i for i in range(data.shape[0])])
        train_idx = [i for i in range(len(np.unique(idx)))]
        val_idx = [i for i in range(len(np.unique(idx)))]
        val_idx.reverse()
        train_data, val_data = data[train_idx], data[val_idx]
        train_targets, val_targets = targets[train_idx], targets[val_idx]
        self.train_trsf, self.test_trsf = data_manager.get_trsf()
        train_dset = DummyDataset1(train_data, train_targets, train_idx, self.train_trsf, data_manager.use_path)
        val_dset = DummyDataset1(val_data, val_targets, val_idx, self.train_trsf, data_manager.use_path)
        
        self.train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        self._network.to(self._device)
        self._stage1_training(self.train_loader, self.val_loader, self.test_loader)
        torch.save(self._network, str(self._cur_task)+'_model.pth')
        del train_dset, val_dset
        del self.train_loader, self.val_loader
        count, all, ratio = self.calculate_sparsity(self._network)
        self.remain_params.append(100-ratio)
        logging.info('remain_params ={}'.format(self.remain_params))
        
        self.build_rehearsal_memory(data_manager, per_class=30)
        buffer_dataset = data_manager.get_dataset([], source='train', mode='test', appendent=self._get_memory())
        self.buffer_loader = DataLoader(buffer_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self._class_means = self.calculate_class_means(self._network, self.buffer_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        # CA
        self._network.fc.backup()
        # if self.save_before_ca:
        #     self.save_checkpoint(self.log_path+'/'+self.model_prefix+'_seed{}_before_ca'.format(self.seed), head_only=self.fix_bcb)
        
        # self._compute_class_mean(data_manager, check_diff=False, oracle=False)
        # if self._cur_task>0 and ca_epochs>0:
        #     # self._stage2_compact_classifier(task_size)
        #     if len(self._multiple_gpus) > 1:
        #         self._network = self._network.module
        
    def _run(self, train_loader, val_loader, test_loader, optimizer, scheduler):
        def grad2vec(model):
            grad_vec = []
            for param in model.parameters():
                if param.grad is not None:
                    grad_vec.append(param.grad.view(-1).detach())
                # for torch2.0.1
                # else:
                #     grad_vec.append(torch.zeros(param.shape).view(-1).to(param.device))
            return torch.cat(grad_vec)
        
        def append_grad_to_vec(vec, model):
            if not isinstance(vec, torch.Tensor):
                raise TypeError('expected torch.Tensor, but got: {}'.format(torch.typename(vec)))
            pointer = 0
            for param in model.parameters():
                if param.grad is not None:
                    num_param = param.numel()
                    if self.args["use_param_grad"]:
                        param.grad.copy_(param.grad + vec[pointer:pointer + num_param].view_as(param).data)
                    else:
                        param.grad.copy_(vec[pointer:pointer + num_param].view_as(param).data)
                    pointer += num_param
        
        lr2 = 0.0001 # vit
        lambd = 0.1
        run_epochs = epochs
        for epoch in range(1, run_epochs+1):
            self._network.train()
            losses, losses_val = 0., 0.
            correct, total = 0, 0
            losses = 0.
            # for i, (_, inputs, targets) in enumerate(train_loader):
            for i, (train_data_batch, val_data_batch) in enumerate(zip(train_loader, val_loader)):
                train_images, train_targets = train_data_batch[1].to(self._device), train_data_batch[2].to(self._device)
                val_images, val_targets = val_data_batch[1].to(self._device), val_data_batch[2].to(self._device)
                train_images_index, val_images_index = train_data_batch[0], val_data_batch[0]
                train_targets -= self._known_classes
                val_targets -= self._known_classes
                # 更新weight，用训练集。只打开weight
                self.switch_mode(mode="finetune")
                # 当前batch内样本的mask
                output = self._network(train_images)['logits']
                weight = self._network.data_mask[train_images_index] # 不映射weight
                if 'resnet' in self.model:
                    weight = torch.sigmoid(weight)
                else:
                    # weight = torch.sigmoid(weight / (1+epoch))
                    weight = torch.sigmoid(weight)
                # weight = 2*torch.sigmoid(weight*(4-epoch))
                # mask = GetSubnetUnstructured.apply(self._network.data_mask.abs()[:self.train_data_num], k) # weight映射成0,1
                # if self._cur_task > 0:
                #     mask1 = torch.ones(self.data_mask_len-self.train_data_num).to(self._device)
                #     mask = torch.cat((mask, mask1), dim=0)
                # weight = mask[train_images_index]
                # pdb.set_trace()
                loss_kd = 0
                if self._cur_task > 0 and 'resnet' in self.model:
                    loss_kd = _KD_loss(output[:,:self._known_classes],self._old_network(train_images)["logits"],2)
                loss = self.calculate_weighted_CE(output[:, self._known_classes:], train_targets, weight.abs())+loss_kd
                # loss = self.calculate_weighted_CE(output, train_targets)+loss_kd
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses += loss.item()
                _, preds = torch.max(output, dim=1)
                correct += preds.eq(train_targets.expand_as(preds)).cpu().sum()
                total += len(train_targets)
                ind = torch.where(preds == train_targets)[0] # 当前batch内预测正确的样本下标

                T_epoch, w = 0, 1
                if 'resnet' in self.model:
                    T_epoch, w = 0, 50
                    if self._cur_task > 0:
                        self.args['update_step'] = 20

                # pdb.set_trace()
                if i % self.args['update_step'] == 0 and epoch >= T_epoch:
                    # 用来计算隐梯度，不更新。delta_z l(z)是对m，theta同时打开时网络回传的梯度
                    # 用验证集算, weight, weight_mask, data_mask均打开
                    self.switch_mode(mode="bilevel")
                    optimizer.zero_grad()
                    output = self._network(val_images)['logits']
                    loss_kd = 0
                    if self._cur_task > 0 and 'resnet' in self.model:
                        loss_kd = _KD_loss(output[:,:self._known_classes],self._old_network(val_images)["logits"],2)
                    loss = F.cross_entropy(output[:, self._known_classes:], val_targets)+loss_kd
                    loss.backward()
                    all_val_unweighted_loss_grad_vec = grad2vec(self._network)
                    if self._network.data_mask.grad is None:
                        tt = torch.zeros(self._network.data_mask.shape[0]).to(self._device)
                        all_val_unweighted_loss_grad_vec = torch.cat((tt, all_val_unweighted_loss_grad_vec), dim=0)
                    
                    # 更新mask，mask_grad用训练集算。打开weight_mask和data_mask
                    self.switch_mode(mode="prune")
                    optimizer.zero_grad()
                    output = self._network(train_images)["logits"]
                    weight = self._network.data_mask[train_images_index] # 不映射weight
                    if 'resnet' in self.model:
                        weight = torch.sigmoid(weight)
                    else:
                        # weight = torch.sigmoid(weight / (epoch+1))
                        weight = torch.sigmoid(weight)
                    # weight = 2*torch.sigmoid(weight*(4-epoch))
                    # mask = GetSubnetUnstructured.apply(self._network.data_mask.abs()[:self.train_data_num], k) # weight映射成0,1
                    # if self._cur_task > 0:
                    #     mask1 = torch.ones(self.data_mask_len-self.train_data_num).to(self._device)
                    #     mask = torch.cat((mask, mask1), dim=0)
                    # weight = mask[train_images_index]
                    loss_kd = 0
                    if self._cur_task > 0 and 'resnet' in self.model:
                        loss_kd = _KD_loss(output[:,:self._known_classes],self._old_network(train_images)["logits"],2)
                    loss_mask = self.calculate_weighted_CE(output[:, self._known_classes:], train_targets, weight.abs())+loss_kd#+lambd*torch.norm(weight, p=1)
                    # loss_mask = self.calculate_weighted_CE(output, train_targets)+loss_kd#+lambd*torch.norm(weight, p=1)
                    # if epoch > 0 and i % 20 == 0:
                    #     print('weight norm = ', torch.norm(weight, p=1))
                    # pdb.set_trace()
                    loss_mask.backward()
                    # pdb.set_trace()
                    # 算与应用隐梯度。delta_z l(z)=param_grad_vec, 
                    mask_train_weighted_loss_grad_vec = grad2vec(self._network)
                    implicit_gradient = -lr2 * (1/self.gamma) * mask_train_weighted_loss_grad_vec * all_val_unweighted_loss_grad_vec
                    del mask_train_weighted_loss_grad_vec
                    del all_val_unweighted_loss_grad_vec

                    # 计算对w的梯度
                    optimizer.zero_grad()
                    output = self._network(val_images)["logits"]
                    loss_kd = 0
                    if self._cur_task > 0 and 'resnet' in self.model:
                        loss_kd = _KD_loss(output[:,:self._known_classes],self._old_network(val_images)["logits"],2)
                    loss_mask = F.cross_entropy(output[:, self._known_classes:], val_targets)+loss_kd
                    loss_mask.backward()
                    mask_val_unweighed_loss_grad_vec = grad2vec(self._network)

                    if not self.args["use_faster_sample_grad"]:
                        # 测试通过，但太慢
                        for j in range(train_images.shape[0]):
                            mask_optimizer.zero_grad()
                            output = self._network(train_images[j:j+1])['logits']
                            loss = F.cross_entropy(output, train_targets[j:j+1])
                            loss.backward()
                            mask_train_unweighed_loss_grad_vec = grad2vec(self._network)/train_images.shape[0]
                            implicit_gradient[train_images_index[j]] = -50*mask_val_unweighed_loss_grad_vec@mask_train_unweighed_loss_grad_vec.t()
                    else:
                        # 加速版
                        output = self._network(train_images)['logits']
                        loss_kd = 0
                        if self._cur_task > 0 and 'resnet' in self.model:
                            output_kl = self._old_network(train_images)["logits"]
                            loss_kd = _KD_loss(output[:,:self._known_classes],output_kl,2)
                        loss = F.cross_entropy(output[:, self._known_classes:], train_targets)+loss_kd
                        optimizer.zero_grad()
                        loss.backward()
                        ori_grad = grad2vec(self._network)
                        loss_ori = loss.item()
                        for j in range(train_images.shape[0]):
                            with torch.no_grad():
                                loss_kd = 0
                                if self._cur_task > 0 and 'resnet' in self.model:
                                    loss_kd = _KD_loss(output[j:j+1,:self._known_classes],output_kl[j:j+1,:],2).item()
                                loss1 = F.cross_entropy(output[j:j+1][:, self._known_classes:], train_targets[j:j+1]).item() + loss_kd
                            mask_train_unweighed_loss_grad_vec = loss1/(train_images.shape[0]*loss_ori) * ori_grad
                            implicit_gradient[train_images_index[j]] = -w*50*mask_val_unweighed_loss_grad_vec@((1/self.gamma)*mask_train_unweighed_loss_grad_vec).t()
                        # pdb.set_trace()
                        del mask_val_unweighed_loss_grad_vec
                        del mask_train_unweighed_loss_grad_vec
                    # assert self._network.data_mask.grad.abs().sum()==0
                    append_grad_to_vec(implicit_gradient, self._network)
                    optimizer.step()
                    # acc1, acc5 = self.accuracy(output, val_targets, topk=(1, 5))
                    losses_val += loss.item()
                    # if self._cur_task > 0:
                    #     pdb.set_trace()

            scheduler.step()
            if epoch%5==0:
                train_acc = self._compute_accuracy(self._network, train_loader)
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.3f}, Test_accy {:.3f}'.format(
                    self._cur_task, epoch, epochs, losses/len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}'.format(
                    self._cur_task, epoch, epochs, losses/len(train_loader))
            logging.info(info)

    def _stage1_training(self, train_loader, val_loader, test_loader):
        '''
        if self._cur_task == 0:
            loaded_dict = torch.load('./dict_0.pkl')
            self._network.load_state_dict(loaded_dict['model_state_dict'])
            self._network.to(self._device)
            return
        '''
        
        base_params = self._network.convnet.parameters()
        base_fc_params = [p for p in self._network.fc.parameters() if p.requires_grad==True]
        head_scale = 1. if 'moco' in self.log_path else 1.
        if not self.fix_bcb:
            base_params = {'params': base_params, 'lr': lrate*self.bcb_lrscale, 'weight_decay': weight_decay}
            base_fc_params = {'params': base_fc_params, 'lr': lrate*head_scale, 'weight_decay': weight_decay}
            network_params = [base_params, base_fc_params]
        else:
            for p in base_params:
                p.requires_grad = False
            network_params = [{'params': base_fc_params, 'lr': lrate*head_scale, 'weight_decay': weight_decay}]
        optimizer = optim.SGD(network_params, lr=lrate, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        
        self._run(train_loader, val_loader, test_loader, optimizer, scheduler)

    def calculate_class_means(self, model, data_loader):
        # calculate class means in train loader
        # class_means = torch.zeros(self._total_classes, self._network.feature_dim)
        print('Calculating class means...')
        model.eval()
        total = [0]*self._total_classes
        all_features = [[] for i in range(self._total_classes)]
        for i, (_, inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            with torch.no_grad():
                image_features = model(inputs)['features']
            vectors = image_features.clone().detach().cpu().numpy()
            for j in range(len(targets)):
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + 1e-8)).T
                total[targets[j]] += 1
                all_features[targets[j]].append(torch.tensor(vectors[j]))
        class_means = []
        for i in range(self._total_classes):
            features = torch.stack(all_features[i], dim=0).sum(dim=0)
            class_means.append(features / total[i])
        class_means = F.normalize(torch.stack(class_means, dim=0), p=2, dim=-1)
        return class_means
    
    def calculate_sparsity(self, model):
        all, count = 0, 0
        for i, v in model.named_modules():
            if hasattr(v, "adj"):
                attn = getattr(v, "adj")
                if not isinstance(attn, int):
                    param = attn.data
                    all += param.nelement()
                    count += torch.sum(param == 0)
                else:
                    return 0, 0, 0
        return count, all, np.round((100*count/all).cpu().numpy(), decimals=2)

    def switch_mode(self, mode='prune'):
        # print(f"#################### Pruning network ####################")
        # print(f"===>>  gradient for weights: None  | training importance scores only")
        assert mode in ["prune", "finetune", "bilevel", "prune_data"]
        if mode == 'finetune':
            self.change_vars("weight_mask", grad=False)
            self.change_vars("weight")
            self._network.data_mask.requires_grad_(False)
        elif mode == 'bilevel':
            self.change_vars("weight_mask")
            self.change_vars("weight")
            self._network.data_mask.requires_grad_(True)
        elif mode == 'prune':
            self.change_vars("weight_mask")
            self.change_vars("weight", grad=False)
            self._network.data_mask.requires_grad_(True)
        else:
            self.change_vars("weight_mask", grad=False)
            self.change_vars("weight", grad=False)
            self._network.data_mask.requires_grad_(True)
        
    def change_vars(self, var_name, freeze_bn=False, grad=True):
        """
        freeze vars. If freeze_bn then only freeze batch_norm params.
        """
        # var_name = qkv.weight, qkv.weight_mask, mlp.fc1.weight, mlp.fc1.weight_mask, mlp.fc2.weight, mlp.fc2.weight_mask
        # assert var_name in ["weight", "bias", "weight_mask"]
        for name, param in self._network.named_parameters():
            if ("mask" in var_name):
                if var_name in name:
                    param.requires_grad_(grad)
            else:
                if self.model == 'vit':
                    # if (("mlp" in name)or("qkv" in name)or("attn.proj" in name))and("mask" not in name):
                    if (("mlp" in name)or("qkv" in name))and("mask" not in name):
                        param.requires_grad_(grad)
                elif 'resnet' in self.model:
                    if ("mask" not in name):
                        param.requires_grad_(grad)
                # if ("fc.weight" == name)or("fc.bias" == name):
                #     param.requires_grad_(grad)

    def calculate_weighted_CE(self, output, targets, weight=None):
        t = 0
        if weight == None:
            weight = torch.ones(len(targets)).to(targets.device)
        # 先类内平均再所有样本平均，考虑了weight映射成0,1的情况
        unique_targets = torch.unique(targets)
        ind_w = torch.where(weight != 0)[0].tolist() # 权重不为0的weight下标
        # for i in unique_targets:
        #     ind_tar = torch.where(targets == i)[0].tolist()
        #     ind = set(ind_tar)&set(ind_w)
        #     tt = 0
        #     if len(ind) != 0:
        #         for j in ind:
        #             tt = tt + weight[j]*F.cross_entropy(output[j], targets[j])
        #         tt = tt / len(ind)
        #     t = t+tt
        # t = t / len(unique_targets)
        # 计算加权平均
        for i in range(output.shape[0]):
            t = t + weight[i]*F.cross_entropy(output[i:i+1], targets[i:i+1])
        # t /= output.shape[0]
        t /= weight[weight!=0].shape[0]
        return t
    
# our
class SLCA(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = FinetuneIncrementalNet(args['convnet_type'], pretrained=True)
        self.log_path = "logs/{}_{}".format(args['model_name'], args['model_postfix'])
        # now = {}
        # for name, param in self._network.convnet.named_parameters():
        #     if 'mask' not in name:
        #         now[name] = param
        # torch.save(now,'now.pth')
        # pdb.set_trace()
        self.model_prefix = args['prefix']
        if 'epochs' in args.keys():
            global epochs
            epochs = args['epochs'] 
        if 'milestones' in args.keys():
            global milestones
            milestones = args['milestones']
        if 'lr' in args.keys():
            global lrate
            lrate = args['lr']
            print('set lr to ', lrate)
        if 'bcb_lrscale' in args.keys():
            self.bcb_lrscale = args['bcb_lrscale']
        else:
            self.bcb_lrscale = 1.0/100
        self.fix_bcb = False
        print('fic_bcb', self.fix_bcb)

        self.seed = args['seed']
        self.task_sizes = []
        
        self.args = args
        self.model = 'vit'
        self.cnn_acc, self.nme_acc = [], []
        self.remain_params = []
        self.gamma = self.args['gamma']
        self.saved_sample_space = [] # 记录每个任务上存的样本的平均字节数，仅对imagenetr统计
        self.mask_ratio_per_task = args["mask_ratio_per_task"]
        self.mask_ratio = 1
        self.upper_bound = False

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))
        # self.save_checkpoint(self.log_path+'/'+self.model_prefix+'_seed{}'.format(self.seed), head_only=self.fix_bcb)
        self._network.fc.recall()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        task_size = data_manager.get_task_size(self._cur_task)
        self.task_sizes.append(task_size)
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.topk = self._total_classes if self._total_classes<5 else 5   
        if self.upper_bound:
            # core50: 400, domainnet: 800
            self._known_classes, self._total_classes = 0, 100
            self._network.update_fc(self._total_classes)
            self.mask_ratio = 1
        else:
            self._network.update_fc(data_manager.get_task_size(self._cur_task))
            # 1 1 2 2 3 3 4 4 5 5
            # self.mask_ratio = self.mask_ratio-(self._cur_task//2+1)/100
            self.mask_ratio = 1-self.mask_ratio_per_task*(self._cur_task+1)
            # self.mask_ratio = 0.4+0.03*(self._cur_task+1)
            # self.mask_ratio = 0.99
            # if self._cur_task == 9:
            #     self.mask_ratio = 0.7
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        enabled, mask_enabled = set(), set()
        if self.model == 'vit':
            # qkv+mlp+proj:84934656 qkv+mlp: 77856768; qkv: 21233664
            for name, param in self._network.named_parameters():
                param.requires_grad_(False)
                if ("fc.head" in name):
                    param.requires_grad_(True)
                # if ("mlp" in name)or("qkv" in name)or("attn.proj" in name):
                if ("mlp" in name)or("qkv" in name):
                    param.requires_grad_(True)
                if param.requires_grad:
                    enabled.add(name)
            for name, m in self._network.named_modules():
                if ("mlp.fc" in name)or("qkv" in name):  
                # if hasattr(m, "weight_mask"):
                    m.set_mask_ratio(self.mask_ratio)
                    mask_enabled.add(name)
        if self._cur_task == 0:
            logging.info('Parameters to be updated: {}'.format(enabled))
            logging.info('Parameters to be masked: {}'.format(mask_enabled))
        if self.args['dataset'] in ['core50', 'domainnet']:
            self._total_classes_test = 50 if self.args['dataset']=='core50' else 200
        else:
            self._total_classes_test = self._total_classes
        # self._total_classes = 200
        data, targets, train_dset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                  source='train', mode='train',
                                                  appendent=self._get_memory(), ret_data=True)
        test_dset = data_manager.get_dataset(np.arange(0, self._total_classes_test), source='test', mode='test')
        dset_name = data_manager.dataset_name.lower()
        # self.try1(data, 10)
        self._network.convnet.reset_data_mask(len(data))
        idx = np.array([i for i in range(data.shape[0])])
        train_idx = [i for i in range(len(np.unique(idx)))]
        val_idx = [i for i in range(len(np.unique(idx)))]
        val_idx.reverse()
        train_data, val_data = data[train_idx], data[val_idx]
        train_targets, val_targets = targets[train_idx], targets[val_idx]
        self.train_trsf, self.test_trsf = data_manager.get_trsf()
        train_dset = DummyDataset1(train_data, train_targets, train_idx, self.train_trsf, data_manager.use_path)
        val_dset = DummyDataset1(val_data, val_targets, val_idx, self.train_trsf, data_manager.use_path)
        
        self.train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        if self.upper_bound:
            val_dset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                    source='train', mode='test',
                                                    ret_data=False)
            self.val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        self._network.to(self._device)
        # self.test(data_manager)
        # pdb.set_trace()
        file_name = './saved_model/'+self.args['dataset']+'/'+str(self._cur_task)+'_model_re_gamma1_5.pth'
        if os.path.exists(file_name):
            self._network = torch.load(file_name).to(self._device)
            logging.info("Loaded trained model : {}".format(file_name))
        else:        
            self._stage1_training(self.train_loader, self.val_loader, self.test_loader)
            # torch.save(self._network, file_name)
            logging.info("Saved model in {}".format(file_name))
        del train_dset, val_dset
        del self.train_loader, self.val_loader
        count, all, ratio = self.calculate_sparsity(self._network)
        self.remain_params.append(100-ratio)
        logging.info('remain_params ={}'.format(self.remain_params))
        
        self.build_rehearsal_memory(data_manager, per_class=self.args["memory_per_class"])
        if self.args['dataset'] in ['imagenet-r']:
            buffer_data, buffer_targets, _ = data_manager.get_dataset([], source='train', mode='test', appendent=self._get_memory(), ret_data=True)
            idx = [i for i in range(buffer_data.shape[0])]
            # imagenetr数据集需要把要保存的样本裁剪再存到本地jpg
            buffer_data = self.crop_and_save(buffer_data, 1, self.args['quality'], self.args['data_save_dir'])
            buffer_dataset = DummyDataset1(buffer_data, buffer_targets, idx, self.test_trsf, use_path=data_manager.use_path)
            self._data_memory = buffer_data
        else:
            buffer_dataset = data_manager.get_dataset([], source='train', mode='test', appendent=self._get_memory())
        self.buffer_loader = DataLoader(buffer_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self._class_means = self.calculate_class_means(self._network, self.buffer_loader)
        # pdb.set_trace()
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        
        # CA
        self._network.fc.backup()
        
    def test(self, data_manager):
        self._network.eval()
        self._known_classes, self._total_classes = 0, 100
        train_dset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                  source='train', mode='train', ret_data=False)
        self.train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        self._class_means = self.calculate_class_means(self._network, self.train_loader)
        y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
        nme_accy = self._evaluate(y_pred, y_true)
        print(nme_accy)
        pdb.set_trace()
    
    def _run_ori(self, train_loader, val_loader, test_loader, optimizer, scheduler):
        run_epochs = epochs
        for epoch in range(0, run_epochs):
            self._network.train()
            losses = 0.
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)['logits']
                cur_targets = torch.where(targets-self._known_classes>=0,targets-self._known_classes,-100)
                loss = F.cross_entropy(logits[:, self._known_classes:], cur_targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            if epoch%2==0:
                train_acc = self._compute_accuracy(self._network, train_loader)
                test_acc = self._compute_accuracy(self._network, test_loader)
                
                self._class_means = self.calculate_class_means(self._network, val_loader)
                y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
                nme_accy = self._evaluate(y_pred, y_true)
                
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.3f}, Test_accy {:.3f}, nme {:.3f}'.format(
                    self._cur_task, epoch, epochs, losses/len(train_loader), train_acc, test_acc, nme_accy['top1'])
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}'.format(
                    self._cur_task, epoch, epochs, losses/len(train_loader))
            logging.info(info)
    
    def _run(self, train_loader, val_loader, test_loader, optimizer, scheduler):
        def grad2vec(model):
            grad_vec = []
            for param in model.parameters():
                if param.grad is not None:
                    grad_vec.append(param.grad.view(-1).detach())
                # for torch2.0.1
                # else:
                #     grad_vec.append(torch.zeros(param.shape).view(-1).to(param.device))
            return torch.cat(grad_vec)
        
        def append_grad_to_vec(vec, model):
            if not isinstance(vec, torch.Tensor):
                raise TypeError('expected torch.Tensor, but got: {}'.format(torch.typename(vec)))
            pointer = 0
            for param in model.parameters():
                if param.grad is not None:
                    num_param = param.numel()
                    if self.args["use_param_grad"]:
                        param.grad.copy_(param.grad + vec[pointer:pointer + num_param].view_as(param).data)
                    else:
                        param.grad.copy_(vec[pointer:pointer + num_param].view_as(param).data)
                    pointer += num_param
        
        lr2 = 1 # same with learning rate
        lambd = 0.1
        run_epochs = epochs
        for epoch in range(1, run_epochs+1):
            self._network.train()
            losses, losses_val = 0., 0.
            correct, total = 0, 0
            losses = 0.
            # for i, (_, inputs, targets) in enumerate(train_loader):
            for i, (train_data_batch, val_data_batch) in enumerate(zip(train_loader, val_loader)):
                train_images, train_targets = train_data_batch[1].to(self._device), train_data_batch[2].to(self._device)
                val_images, val_targets = val_data_batch[1].to(self._device), val_data_batch[2].to(self._device)
                train_images_index, val_images_index = train_data_batch[0], val_data_batch[0]
                # train_targets -= self._known_classes
                # val_targets -= self._known_classes
                # 更新weight，用训练集。只打开weight
                self.switch_mode(mode="finetune")
                # 当前batch内样本的mask
                output = self._network(train_images)['logits']
                weight = self._network.convnet.data_mask[train_images_index] # 不映射weight
                if 'resnet' in self.model:
                    weight = torch.sigmoid(weight)
                else:
                    # weight = torch.sigmoid(weight / (1+epoch))
                    weight = torch.sigmoid(weight)
                # weight = 2*torch.sigmoid(weight*(4-epoch))
                # mask = GetSubnetUnstructured.apply(self._network.data_mask.abs()[:self.train_data_num], k) # weight映射成0,1
                # if self._cur_task > 0:
                #     mask1 = torch.ones(self.data_mask_len-self.train_data_num).to(self._device)
                #     mask = torch.cat((mask, mask1), dim=0)
                # weight = mask[train_images_index]
                # pdb.set_trace()
                loss_kd = 0
                if self._cur_task > 0 and 'resnet' in self.model:
                    loss_kd = _KD_loss(output[:,:self._known_classes],self._old_network(train_images)["logits"],2)
                loss = self.calculate_weighted_CE(output, train_targets, weight.abs())+loss_kd
                # loss = self.calculate_weighted_CE(output, train_targets)+loss_kd
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses += loss.item()
                _, preds = torch.max(output, dim=1)
                correct += preds.eq(train_targets.expand_as(preds)).cpu().sum()
                total += len(train_targets)
                ind = torch.where(preds == train_targets)[0] # 当前batch内预测正确的样本下标

                T_epoch, w = 0, 1
                if 'resnet' in self.model:
                    T_epoch, w = 0, 50
                    if self._cur_task > 0:
                        self.args['update_step'] = 20

                # pdb.set_trace()
                # if i % (self.args['update_step']*(self._cur_task*0.5+1)) == 0 and epoch >= T_epoch:
                if i % (self.args['update_step']) == 0 and epoch >= T_epoch:
                    # 用来计算隐梯度，不更新。delta_z l(z)是对m，theta同时打开时网络回传的梯度
                    # 用验证集算, weight, weight_mask, data_mask均打开
                    self.switch_mode(mode="bilevel")
                    optimizer.zero_grad()
                    output = self._network(val_images)['logits']
                    loss_kd = 0
                    if self._cur_task > 0 and 'resnet' in self.model:
                        loss_kd = _KD_loss(output[:,:self._known_classes],self._old_network(val_images)["logits"],2)
                    loss = F.cross_entropy(output, val_targets)+loss_kd
                    loss.backward()
                    all_val_unweighted_loss_grad_vec = grad2vec(self._network)
                    if self._network.convnet.data_mask.grad is None:
                        tt = torch.zeros(self._network.convnet.data_mask.shape[0]).to(self._device)
                        all_val_unweighted_loss_grad_vec = torch.cat((tt, all_val_unweighted_loss_grad_vec), dim=0)
                    
                    # 更新mask，mask_grad用训练集算。打开weight_mask和data_mask
                    self.switch_mode(mode="prune")
                    optimizer.zero_grad()
                    output = self._network(train_images)["logits"]
                    weight = self._network.convnet.data_mask[train_images_index] # 不映射weight
                    if 'resnet' in self.model:
                        weight = torch.sigmoid(weight)
                    else:
                        # weight = torch.sigmoid(weight / (epoch+1))
                        weight = torch.sigmoid(weight)
                    # weight = 2*torch.sigmoid(weight*(4-epoch))
                    # mask = GetSubnetUnstructured.apply(self._network.data_mask.abs()[:self.train_data_num], k) # weight映射成0,1
                    # if self._cur_task > 0:
                    #     mask1 = torch.ones(self.data_mask_len-self.train_data_num).to(self._device)
                    #     mask = torch.cat((mask, mask1), dim=0)
                    # weight = mask[train_images_index]
                    loss_kd = 0
                    if self._cur_task > 0 and 'resnet' in self.model:
                        loss_kd = _KD_loss(output[:,:self._known_classes],self._old_network(train_images)["logits"],2)
                    loss_mask = self.calculate_weighted_CE(output, train_targets, weight.abs())+loss_kd#+lambd*torch.norm(weight, p=1)
                    # loss_mask = self.calculate_weighted_CE(output, train_targets)+loss_kd#+lambd*torch.norm(weight, p=1)
                    # if epoch > 0 and i % 20 == 0:
                    #     print('weight norm = ', torch.norm(weight, p=1))
                    # pdb.set_trace()
                    loss_mask.backward()
                    # pdb.set_trace()
                    # 算与应用隐梯度。delta_z l(z)=param_grad_vec, 
                    mask_train_weighted_loss_grad_vec = grad2vec(self._network)
                    implicit_gradient = -lr2 * (1/self.gamma) * mask_train_weighted_loss_grad_vec * all_val_unweighted_loss_grad_vec
                    # print(implicit_gradient.abs().min(),' ',implicit_gradient.abs().max())
                    del mask_train_weighted_loss_grad_vec
                    del all_val_unweighted_loss_grad_vec

                    # 计算对w的梯度
                    optimizer.zero_grad()
                    output = self._network(val_images)["logits"]
                    loss_kd = 0
                    if self._cur_task > 0 and 'resnet' in self.model:
                        loss_kd = _KD_loss(output[:,:self._known_classes],self._old_network(val_images)["logits"],2)
                    loss_mask = F.cross_entropy(output, val_targets)+loss_kd
                    loss_mask.backward()
                    mask_val_unweighed_loss_grad_vec = grad2vec(self._network)

                    if not self.args["use_faster_sample_grad"]:
                        # 测试通过，但太慢
                        for j in range(train_images.shape[0]):
                            mask_optimizer.zero_grad()
                            output = self._network(train_images[j:j+1])['logits']
                            loss = F.cross_entropy(output, train_targets[j:j+1])
                            loss.backward()
                            mask_train_unweighed_loss_grad_vec = grad2vec(self._network)/train_images.shape[0]
                            implicit_gradient[train_images_index[j]] = -50*mask_val_unweighed_loss_grad_vec@mask_train_unweighed_loss_grad_vec.t()
                    else:
                        # 加速版
                        output = self._network(train_images)['logits']
                        loss_kd = 0
                        if self._cur_task > 0 and 'resnet' in self.model:
                            output_kl = self._old_network(train_images)["logits"]
                            loss_kd = _KD_loss(output[:,:self._known_classes],output_kl,2)
                        loss = F.cross_entropy(output, train_targets)+loss_kd
                        optimizer.zero_grad()
                        loss.backward()
                        ori_grad = grad2vec(self._network)
                        loss_ori = loss.item()
                        for j in range(train_images.shape[0]):
                            with torch.no_grad():
                                loss_kd = 0
                                if self._cur_task > 0 and 'resnet' in self.model:
                                    loss_kd = _KD_loss(output[j:j+1,:self._known_classes],output_kl[j:j+1,:],2).item()
                                loss1 = F.cross_entropy(output[j:j+1], train_targets[j:j+1]).item() + loss_kd
                            mask_train_unweighed_loss_grad_vec = loss1/(train_images.shape[0]*loss_ori) * ori_grad
                            implicit_gradient[train_images_index[j]] = -w*50*mask_val_unweighed_loss_grad_vec@((1/self.gamma)*mask_train_unweighed_loss_grad_vec).t()
                        # pdb.set_trace()
                        # print(' ',mask_val_unweighed_loss_grad_vec.max(),' ',mask_train_unweighed_loss_grad_vec.abs().max(),' ',implicit_gradient.abs().max())
                        # print(' ',mask_val_unweighed_loss_grad_vec.abs().max(),' ',mask_train_unweighed_loss_grad_vec.abs().max(),' ',implicit_gradient[:5000].abs().max())
                        # print(implicit_gradient[:5000].max(),' ',implicit_gradient[:5000].min())
                        # print(self._network.convnet.data_mask.data.max(),' ',self._network.convnet.data_mask.data.min(),' ',self._network.convnet.data_mask.data.sum()/5000)
                        # print('\n')
                        del mask_val_unweighed_loss_grad_vec
                        del mask_train_unweighed_loss_grad_vec
                    # assert self._network.convnet.data_mask.grad.abs().sum()==0
                    append_grad_to_vec(implicit_gradient, self._network)
                    optimizer.step()
                    # acc1, acc5 = self.accuracy(output, val_targets, topk=(1, 5))
                    losses_val += loss.item()
                    # if self._cur_task > 0:
                    #     pdb.set_trace()

            scheduler.step()
            if epoch%5==0:
                train_acc = self._compute_accuracy(self._network, train_loader)
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.3f}, Test_accy {:.3f}'.format(
                    self._cur_task, epoch, epochs, losses/len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}'.format(
                    self._cur_task, epoch, epochs, losses/len(train_loader))
            logging.info(info)

    def _stage1_training(self, train_loader, val_loader, test_loader):
        '''
        if self._cur_task == 0:
            loaded_dict = torch.load('./dict_0.pkl')
            self._network.load_state_dict(loaded_dict['model_state_dict'])
            self._network.to(self._device)
            return
        '''
        
        base_params = self._network.convnet.parameters()
        base_fc_params = [p for p in self._network.fc.parameters() if p.requires_grad==True]
        head_scale = 1. if 'moco' in self.log_path else 1.
        # lrate*self.bcb_lrscale lrate*head_scale
        base_params = {'params': base_params, 'lr': self.args['lr_base'], 'weight_decay': weight_decay}
        base_fc_params = {'params': base_fc_params, 'lr': self.args['lr_fc'], 'weight_decay': weight_decay}
        network_params = [base_params, base_fc_params]
        optimizer = optim.SGD(network_params, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
        
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        
        if self.upper_bound:
            self._run_ori(train_loader, val_loader, test_loader, optimizer, scheduler)
        else:
            self._run(train_loader, val_loader, test_loader, optimizer, scheduler)

    def calculate_class_means(self, model, data_loader):
        # calculate class means in train loader
        # class_means = torch.zeros(self._total_classes, self._network.feature_dim)
        print('Calculating class means...')
        model.eval()
        total = [0]*self._total_classes
        all_features = [[] for i in range(self._total_classes)]
        for i, (_, inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            with torch.no_grad():
                image_features = model(inputs)['features']
            vectors = image_features.clone().detach().cpu().numpy()
            for j in range(len(targets)):
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + 1e-8)).T
                total[targets[j]] += 1
                all_features[targets[j]].append(torch.tensor(vectors[j]))
        class_means = []
        for i in range(self._total_classes):
            features = torch.stack(all_features[i], dim=0).sum(dim=0)
            class_means.append(features / total[i])
        class_means = F.normalize(torch.stack(class_means, dim=0), p=2, dim=-1)
        return class_means
    
    def calculate_sparsity(self, model):
        all, count = 0, 0
        for i, v in model.named_modules():
            if hasattr(v, "adj"):
                attn = getattr(v, "adj")
                if not isinstance(attn, int):
                    param = attn.data
                    all += param.nelement()
                    count += torch.sum(param == 0)
                else:
                    return 0, 0, 0
        return count, all, np.round((100*count/all).cpu().numpy(), decimals=2)

    def switch_mode(self, mode='prune'):
        # print(f"#################### Pruning network ####################")
        # print(f"===>>  gradient for weights: None  | training importance scores only")
        assert mode in ["prune", "finetune", "bilevel", "prune_data"]
        if mode == 'finetune':
            self.change_vars("weight_mask", grad=False)
            self.change_vars("weight")
            self._network.convnet.data_mask.requires_grad_(False)
        elif mode == 'bilevel':
            self.change_vars("weight_mask")
            self.change_vars("weight")
            self._network.convnet.data_mask.requires_grad_(True)
        elif mode == 'prune':
            self.change_vars("weight_mask")
            self.change_vars("weight", grad=False)
            self._network.convnet.data_mask.requires_grad_(True)
        else:
            self.change_vars("weight_mask", grad=False)
            self.change_vars("weight", grad=False)
            self._network.convnet.data_mask.requires_grad_(True)
        
    def change_vars(self, var_name, freeze_bn=False, grad=True):
        """
        freeze vars. If freeze_bn then only freeze batch_norm params.
        """
        # var_name = qkv.weight, qkv.weight_mask, mlp.fc1.weight, mlp.fc1.weight_mask, mlp.fc2.weight, mlp.fc2.weight_mask
        # assert var_name in ["weight", "bias", "weight_mask"]
        for name, param in self._network.named_parameters():
            if ("mask" in var_name):
                if var_name in name:
                    param.requires_grad_(grad)
            else:
                if self.model == 'vit':
                    # if (("mlp" in name)or("qkv" in name)or("attn.proj" in name))and("mask" not in name):
                    if (("mlp" in name)or("qkv" in name))and("mask" not in name):
                        param.requires_grad_(grad)
                elif 'resnet' in self.model:
                    if ("mask" not in name):
                        param.requires_grad_(grad)
                # if ("fc.weight" == name)or("fc.bias" == name):
                #     param.requires_grad_(grad)

    def calculate_weighted_CE(self, output, targets, weight=None):
        t = 0
        if weight == None:
            weight = torch.ones(len(targets)).to(targets.device)
        # 先类内平均再所有样本平均，考虑了weight映射成0,1的情况
        unique_targets = torch.unique(targets)
        ind_w = torch.where(weight != 0)[0].tolist() # 权重不为0的weight下标
        # for i in unique_targets:
        #     ind_tar = torch.where(targets == i)[0].tolist()
        #     ind = set(ind_tar)&set(ind_w)
        #     tt = 0
        #     if len(ind) != 0:
        #         for j in ind:
        #             tt = tt + weight[j]*F.cross_entropy(output[j], targets[j])
        #         tt = tt / len(ind)
        #     t = t+tt
        # t = t / len(unique_targets)
        # 计算加权平均
        for i in range(output.shape[0]):
            t = t + weight[i]*F.cross_entropy(output[i:i+1], targets[i:i+1])
        # t /= output.shape[0]
        t /= weight[weight!=0].shape[0]
        return t
    
    def crop_and_save(self, file_name, ratio, quality, save_dir):
        # file_name: 样本路径表。下面算裁剪前和裁剪后的图像大小
        # train_trsf = [transforms.RandomResizedCrop(224),]
        new_file_name = []
        ori_space, space, file_num = 0, 0, 0
        for img_ori_path in file_name:
            if 'data_buffer' not in img_ori_path:
                part = img_ori_path.split('/')
                new_name = os.path.join(save_dir,part[-2]+'_'+part[-1])
                
                img_size = Image.open(img_ori_path).size
                width, height = img_size[0]*ratio, img_size[1]*ratio
                train_trsf = [transforms.CenterCrop((height, width)),]
                # train_trsf = [transforms.CenterCrop((375, 375)),]
                trsf = transforms.Compose([*train_trsf])
                img =  trsf(pil_loader(img_ori_path))
                # train_trsf = [
                #     transforms.CenterCrop((height, width)),
                #     # transforms.RandomResizedCrop((height, width)),
                #     # transforms.Resize((400, 400)),
                # ]
                # trsf = transforms.Compose([*train_trsf])
                # img = trsf(img_ori)
                # img_ori = pil_loader(img_ori_path)
                # img = img_ori
                # if img_size[0] > 400 and img_size[1] > 400:
                #     img = trsf(img_ori)
                    
                # 用PIL自带的裁剪
                # img_ori = Image.open(img_ori_path)
                # img_size = img_ori.size
                # width, height = img_size[0]*ratio, img_size[1]*ratio
                # begin, end = img_size[0]*(1-ratio)/2, img_size[1]*(1-ratio)/2
                # img = img_ori.crop((begin, end, begin+width, end+height))
                
                img.save(new_name, quality=quality)
                space += os.path.getsize(new_name)
                ori_space += os.path.getsize(img_ori_path)
                file_num += 1
                new_file_name.append(new_name)
            else:
                new_file_name.append(img_ori_path)
        self.saved_sample_space.append(space/file_num)
        logging.info('curr task saved img space/ori img space | all avg saved img space:{}/{} | {}'.format(space, ori_space, sum(self.saved_sample_space)/len(self.saved_sample_space)))
        return np.array(new_file_name)

    def try1(self, file_name, quality):
        # width, height = set(), set()
        # all_size = set()
        # for img_ori_path in file_name:
        #     img = Image.open(img_ori_path)
        #     # pdb.set_trace()
        #     width.add(img.size[0])
        #     height.add(img.size[1])
        #     all_size.add(img.size)
        # pdb.set_trace()
        # name: 样本路径。下面算裁剪前和裁剪后的图像大小
        # ratio=0.6:19130.67, space/ori_space=0.265
        # ratio=0.7:24198.42, space/ori_space=0.336
        # ratio=0.8:29200.69, space/ori_space=0.405
        # avg ori space = 72109
        # ratio=0.9, quality=90, 55526.86
        # ratio=0.9, quality=85, 45102
        # ratio=0.8, quality=95, 65221
        # ratio=0.7, quality=95, 53629
        # ratio=0.6, quality=95, 42081
        # ratio=1, quality=95, 82074, 10样本
        # ratio=1, quality=90, 61375
        # ratio=1, quality=85, 53075  16样本
        # ratio=1, quality=80, 45411
        # ratio=1, quality=75, 37677.5 20样本
        # ratio=1, quality=55, 27953  30样本
        # ratio=1, quality=35, 21678.65, 40样本
        # ratio=1, quality=15, 13598.44, 64样本
        # ratio=1, quality=10, 11140, 80样本
        # ratio=1, quality=5, 8322, 105样本
        # domainnet: after:34276 before:34355
        # core50: 
        # 77856768
        path = '/home/myh/ATS-PyCIL-master_22/data_buffer/domainnet'
        ori_space, space = 0, 0
        num = 0
        ratio = 1
        wid, high, all_area = [], [], 0
        for img_ori_path in file_name:
            part = img_ori_path.split('/')
            new_name = os.path.join(path,part[-2]+'_'+part[-1])
            img_ori = pil_loader(img_ori_path)
            img_size = Image.open(img_ori_path).size
            # img_ori = Image.open(img_ori_path)
            # img_size = img_ori.size
            
            wid.append(img_size[0])
            high.append(img_size[1])
            # # all_area += img_size[0]*img_size[1]
            
            # # 用深度学习裁剪，大小对不上
            width, height = img_size[0]*ratio, img_size[1]*ratio
            # width, height = 224, 224
            train_trsf = [
                transforms.CenterCrop((height, width)),
                # transforms.RandomResizedCrop((height, width)),
                # transforms.Resize((400, 400)),
            ]
            trsf = transforms.Compose([*train_trsf])
            img = trsf(img_ori)
            
            # if img_size[0] > 400 and img_size[1] > 400:
            #     img = trsf(img_ori)
            # else:
            #     img = img_ori
            
            # 用PIL自带的裁剪
            # width, height = img_size[0]*ratio, img_size[1]*ratio
            # begin, end = img_size[0]*(1-ratio)/2, img_size[1]*(1-ratio)/2
            # img = img_ori.crop((begin, end, begin+width, end+height))
            
            # pdb.set_trace()
            img.save(new_name, quality=quality)
            space += os.path.getsize(new_name)
            # ori_space += os.path.getsize(img_ori_path)
            num += 1
            # pdb.set_trace()
        num = 1
        print(space/num)
        pdb.set_trace()

# 模块消融实验,change from SLCA_our
class SLCA1(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = FinetuneIncrementalNet(args['convnet_type'], pretrained=True)
        self.log_path = "logs/{}_{}".format(args['model_name'], args['model_postfix'])
        # now = {}
        # for name, param in self._network.convnet.named_parameters():
        #     if 'mask' not in name:
        #         now[name] = param
        # torch.save(now,'now.pth')
        # pdb.set_trace()
        self.model_prefix = args['prefix']
        if 'epochs' in args.keys():
            global epochs
            epochs = args['epochs'] 
        if 'milestones' in args.keys():
            global milestones
            milestones = args['milestones']
        if 'lr' in args.keys():
            global lrate
            lrate = args['lr']
            print('set lr to ', lrate)
        if 'bcb_lrscale' in args.keys():
            self.bcb_lrscale = args['bcb_lrscale']
        else:
            self.bcb_lrscale = 1.0/100
        self.fix_bcb = False
        print('fic_bcb', self.fix_bcb)

        self.seed = args['seed']
        self.task_sizes = []
        
        self.args = args
        self.model = 'vit'
        self.cnn_acc, self.nme_acc = [], []
        self.remain_params = []
        self.gamma = self.args['gamma']
        self.saved_sample_space = [] # 记录每个任务上存的样本的平均字节数，仅对imagenetr统计
        self.mask_ratio_per_task = args["mask_ratio_per_task"]
        self.upper_bound = False

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))
        # self.save_checkpoint(self.log_path+'/'+self.model_prefix+'_seed{}'.format(self.seed), head_only=self.fix_bcb)
        self._network.fc.recall()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        task_size = data_manager.get_task_size(self._cur_task)
        self.task_sizes.append(task_size)
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.topk = self._total_classes if self._total_classes<5 else 5   
        if self.upper_bound:
            # core50: 400
            self._known_classes, self._total_classes = 0, 400
            self._network.update_fc(self._total_classes)
            self.mask_ratio = 1
        else:
            self._network.update_fc(data_manager.get_task_size(self._cur_task))
            self.mask_ratio = 1-self.mask_ratio_per_task*(self._cur_task+1)
            self.mask_ratio = 1
            # self.mask_ratio = 0.4+0.03*(self._cur_task+1)
            # if self._cur_task == 9:
            #     self.mask_ratio = 0.7
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        enabled, mask_enabled = set(), set()
        if self.model == 'vit':
            # qkv+mlp+proj:84934656 qkv+mlp: 77856768; qkv: 21233664
            for name, param in self._network.named_parameters():
                param.requires_grad_(False)
                # if ("fc.heads."+str(self._cur_task) in name):
                if ("fc.heads" in name):
                    param.requires_grad_(True)
                # if ("mlp" in name)or("qkv" in name)or("attn.proj" in name):
                if ("mlp" in name)or("qkv" in name):
                    param.requires_grad_(True)
                if param.requires_grad:
                    enabled.add(name)
            for name, m in self._network.named_modules():
                if ("mlp.fc" in name)or("qkv" in name):  
                # if hasattr(m, "weight_mask"):
                    m.set_mask_ratio(self.mask_ratio)
                    mask_enabled.add(name)
        if self._cur_task == 0:
            logging.info('Parameters to be updated: {}'.format(enabled))
            logging.info('Parameters to be masked: {}'.format(mask_enabled))
        if self.args['dataset'] in ['core50', 'domainnet']:
            self._total_classes_test = 50 if self.args['dataset']=='core50' else 200
        else:
            self._total_classes_test = self._total_classes
        
        data, targets, train_dset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                  source='train', mode='train',
                                                  appendent=self._get_memory(), ret_data=True)
        test_dset = data_manager.get_dataset(np.arange(0, self._total_classes_test), source='test', mode='test')
        dset_name = data_manager.dataset_name.lower()
        
        self._network.convnet.reset_data_mask(len(data))
        idx = np.array([i for i in range(data.shape[0])])
        train_idx = [i for i in range(len(np.unique(idx)))]
        val_idx = [i for i in range(len(np.unique(idx)))]
        val_idx.reverse()
        train_data, val_data = data[train_idx], data[val_idx]
        train_targets, val_targets = targets[train_idx], targets[val_idx]
        self.train_trsf, self.test_trsf = data_manager.get_trsf()
        train_dset = DummyDataset1(train_data, train_targets, train_idx, self.train_trsf, data_manager.use_path)
        val_dset = DummyDataset1(val_data, val_targets, val_idx, self.train_trsf, data_manager.use_path)
        
        self.train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        if self.upper_bound:
            val_dset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                    source='train', mode='test',
                                                    ret_data=False)
            self.val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        self._network.to(self._device)
        # self.test(data_manager)
        # pdb.set_trace()
        file_name = './saved_model/'+self.args['dataset']+'/'+str(self._cur_task)+'_model_re_gamma1_5.pth'
        if os.path.exists(file_name):
            self._network = torch.load(file_name).to(self._device)
            logging.info("Loaded trained model : {}".format(file_name))
        else:        
            self._stage1_training(self.train_loader, self.val_loader, self.test_loader)
            # torch.save(self._network, file_name)
            logging.info("Saved model in {}".format(file_name))
        del train_dset, val_dset, self.train_loader, self.val_loader
        count, all, ratio = self.calculate_sparsity(self._network)
        self.remain_params.append(100-ratio)
        logging.info('remain_params ={}'.format(self.remain_params))
        
        self.build_rehearsal_memory(data_manager, per_class=self.args["memory_per_class"])
        if self.args['dataset'] in ['imagenet-r']:
            buffer_data, buffer_targets, _ = data_manager.get_dataset([], source='train', mode='test', appendent=self._get_memory(), ret_data=True)
            idx = [i for i in range(buffer_data.shape[0])]
            # imagenetr数据集需要把要保存的样本裁剪再存到本地jpg
            buffer_data = self.crop_and_save(buffer_data, 1, self.args['quality'], self.args['data_save_dir'])
            buffer_dataset = DummyDataset1(buffer_data, buffer_targets, idx, self.test_trsf, use_path=data_manager.use_path)
            self._data_memory = buffer_data
        else:
            buffer_dataset = data_manager.get_dataset([], source='train', mode='test', appendent=self._get_memory())
        self.buffer_loader = DataLoader(buffer_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self._class_means = self.calculate_class_means(self._network, self.buffer_loader)
        # pdb.set_trace()
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        # CA
        self._network.fc.backup()
        
    def test(self, data_manager):
        self._network.eval()
        self._known_classes, self._total_classes = 0, 100
        train_dset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                  source='train', mode='train', ret_data=False)
        self.train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        self._class_means = self.calculate_class_means(self._network, self.train_loader)
        y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
        nme_accy = self._evaluate(y_pred, y_true)
        print(nme_accy)
        pdb.set_trace()
    
    def _run_ori(self, train_loader, val_loader, test_loader, optimizer, scheduler):
        run_epochs = epochs
        self.switch_mode(mode="finetune")
        self._network.convnet.data_mask.requires_grad_(True)
        for epoch in range(0, run_epochs):
            self._network.train()
            losses = 0.
            for i, (train_images_index, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)['logits']
                
                # loss = F.cross_entropy(logits[:, self._known_classes:], targets-self._known_classes)
                # loss = F.cross_entropy(logits[:, self._known_classes:], targets-self._known_classes)
                weight = self._network.convnet.data_mask[train_images_index]
                weight = torch.sigmoid(weight)
                # loss = self.calculate_weighted_CE(logits[:, self._known_classes:], targets-self._known_classes, weight.abs())
                loss = self.calculate_weighted_CE(logits, targets)#, weight.abs())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            if epoch%5==0:
                train_acc = self._compute_accuracy(self._network, train_loader)
                test_acc = self._compute_accuracy(self._network, test_loader)
                
                # self._class_means = self.calculate_class_means(self._network, val_loader)
                # y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
                # nme_accy = self._evaluate(y_pred, y_true)
                
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.3f}, Test_accy {:.3f}'.format(
                    self._cur_task, epoch, epochs, losses/len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}'.format(
                    self._cur_task, epoch, epochs, losses/len(train_loader))
            logging.info(info)
    
    def _run2(self, train_loader, val_loader, test_loader, optimizer, scheduler):
        def grad2vec(model):
            grad_vec = []
            for param in model.parameters():
                if param.grad is not None:
                    grad_vec.append(param.grad.view(-1).detach())
                # for torch2.0.1
                # else:
                #     grad_vec.append(torch.zeros(param.shape).view(-1).to(param.device))
            return torch.cat(grad_vec)
        
        def append_grad_to_vec(vec, model):
            if not isinstance(vec, torch.Tensor):
                raise TypeError('expected torch.Tensor, but got: {}'.format(torch.typename(vec)))
            pointer = 0
            for param in model.parameters():
                if param.grad is not None:
                    num_param = param.numel()
                    if self.args["use_param_grad"]:
                        param.grad.copy_(param.grad + vec[pointer:pointer + num_param].view_as(param).data)
                    else:
                        param.grad.copy_(vec[pointer:pointer + num_param].view_as(param).data)
                    pointer += num_param
        
        lr2 = 1 # same with learning rate
        lambd = 0.1
        run_epochs = epochs
        for epoch in range(1, run_epochs+1):
            self._network.train()
            losses, losses_val = 0., 0.
            correct, total = 0, 0
            losses = 0.
            # for i, (_, inputs, targets) in enumerate(train_loader):
            for i, (train_data_batch, val_data_batch) in enumerate(zip(train_loader, val_loader)):
                train_images, train_targets = train_data_batch[1].to(self._device), train_data_batch[2].to(self._device)
                val_images, val_targets = val_data_batch[1].to(self._device), val_data_batch[2].to(self._device)
                train_images_index, val_images_index = train_data_batch[0], val_data_batch[0]
                # train_targets -= self._known_classes
                # val_targets -= self._known_classes
                # 更新weight，用训练集。只打开weight
                self.switch_mode(mode="finetune")
                # 当前batch内样本的mask
                output = self._network(train_images)['logits']
                weight = self._network.convnet.data_mask[train_images_index] # 不映射weight
                weight = torch.sigmoid(weight)
                # weight = 2*torch.sigmoid(weight*(4-epoch))
                # mask = GetSubnetUnstructured.apply(self._network.data_mask.abs()[:self.train_data_num], k) # weight映射成0,1
                # if self._cur_task > 0:
                #     mask1 = torch.ones(self.data_mask_len-self.train_data_num).to(self._device)
                #     mask = torch.cat((mask, mask1), dim=0)
                # weight = mask[train_images_index]
                # pdb.set_trace()
                loss = F.cross_entropy(output, train_targets)
                # loss = self.calculate_weighted_CE(output, train_targets)+loss_kd
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses += loss.item()
                _, preds = torch.max(output, dim=1)
                correct += preds.eq(train_targets.expand_as(preds)).cpu().sum()
                total += len(train_targets)
                ind = torch.where(preds == train_targets)[0] # 当前batch内预测正确的样本下标

                T_epoch, w = 0, 1
                if 'resnet' in self.model:
                    T_epoch, w = 0, 50
                    if self._cur_task > 0:
                        self.args['update_step'] = 20

                # pdb.set_trace()
                if i % self.args['update_step'] == 0 and epoch >= T_epoch:
                    # 用来计算隐梯度，不更新。delta_z l(z)是对m，theta同时打开时网络回传的梯度
                    # 用验证集算, weight, weight_mask, data_mask均打开
                    self.switch_mode(mode="bilevel")
                    optimizer.zero_grad()
                    output = self._network(val_images)['logits']
                    loss = F.cross_entropy(output, val_targets)
                    loss.backward()
                    all_val_unweighted_loss_grad_vec = grad2vec(self._network)
                    # if self._network.convnet.data_mask.grad is None:
                    #     tt = torch.zeros(self._network.convnet.data_mask.shape[0]).to(self._device)
                    #     all_val_unweighted_loss_grad_vec = torch.cat((tt, all_val_unweighted_loss_grad_vec), dim=0)
                    
                    # 更新mask，mask_grad用训练集算。打开weight_mask和data_mask
                    self.switch_mode(mode="prune")
                    optimizer.zero_grad()
                    output = self._network(train_images)["logits"]
                    weight = self._network.convnet.data_mask[train_images_index] # 不映射weight
                    weight = torch.sigmoid(weight)
                    # weight = 2*torch.sigmoid(weight*(4-epoch))
                    # mask = GetSubnetUnstructured.apply(self._network.data_mask.abs()[:self.train_data_num], k) # weight映射成0,1
                    # if self._cur_task > 0:
                    #     mask1 = torch.ones(self.data_mask_len-self.train_data_num).to(self._device)
                    #     mask = torch.cat((mask, mask1), dim=0)
                    # weight = mask[train_images_index]
                    loss_mask = F.cross_entropy(output, train_targets)#+lambd*torch.norm(weight, p=1)
                    # loss_mask = self.calculate_weighted_CE(output, train_targets)+loss_kd#+lambd*torch.norm(weight, p=1)
                    # if epoch > 0 and i % 20 == 0:
                    #     print('weight norm = ', torch.norm(weight, p=1))
                    # pdb.set_trace()
                    loss_mask.backward()
                    # pdb.set_trace()
                    # 算与应用隐梯度。delta_z l(z)=param_grad_vec, 
                    mask_train_weighted_loss_grad_vec = grad2vec(self._network)
                    # pdb.set_trace()
                    implicit_gradient = -lr2 * (1/self.gamma) * mask_train_weighted_loss_grad_vec * all_val_unweighted_loss_grad_vec
                    # print(implicit_gradient.abs().min(),' ',implicit_gradient.abs().max())
                    del mask_train_weighted_loss_grad_vec
                    del all_val_unweighted_loss_grad_vec

                    # 计算对w的梯度
                    # optimizer.zero_grad()
                    # output = self._network(val_images)["logits"]
                    # loss_kd = 0
                    # if self._cur_task > 0 and 'resnet' in self.model:
                    #     loss_kd = _KD_loss(output[:,:self._known_classes],self._old_network(val_images)["logits"],2)
                    # loss_mask = F.cross_entropy(output, val_targets)+loss_kd
                    # loss_mask.backward()
                    # mask_val_unweighed_loss_grad_vec = grad2vec(self._network)

                    # if not self.args["use_faster_sample_grad"]:
                    #     # 测试通过，但太慢
                    #     for j in range(train_images.shape[0]):
                    #         mask_optimizer.zero_grad()
                    #         output = self._network(train_images[j:j+1])['logits']
                    #         loss = F.cross_entropy(output, train_targets[j:j+1])
                    #         loss.backward()
                    #         mask_train_unweighed_loss_grad_vec = grad2vec(self._network)/train_images.shape[0]
                    #         implicit_gradient[train_images_index[j]] = -50*mask_val_unweighed_loss_grad_vec@mask_train_unweighed_loss_grad_vec.t()
                    # else:
                    #     # 加速版
                    #     output = self._network(train_images)['logits']
                    #     loss_kd = 0
                    #     if self._cur_task > 0 and 'resnet' in self.model:
                    #         output_kl = self._old_network(train_images)["logits"]
                    #         loss_kd = _KD_loss(output[:,:self._known_classes],output_kl,2)
                    #     loss = F.cross_entropy(output, train_targets)+loss_kd
                    #     optimizer.zero_grad()
                    #     loss.backward()
                    #     ori_grad = grad2vec(self._network)
                    #     loss_ori = loss.item()
                    #     for j in range(train_images.shape[0]):
                    #         with torch.no_grad():
                    #             loss_kd = 0
                    #             if self._cur_task > 0 and 'resnet' in self.model:
                    #                 loss_kd = _KD_loss(output[j:j+1,:self._known_classes],output_kl[j:j+1,:],2).item()
                    #             loss1 = F.cross_entropy(output[j:j+1], train_targets[j:j+1]).item() + loss_kd
                    #         mask_train_unweighed_loss_grad_vec = loss1/(train_images.shape[0]*loss_ori) * ori_grad
                    #         implicit_gradient[train_images_index[j]] = -w*50*mask_val_unweighed_loss_grad_vec@((1/self.gamma)*mask_train_unweighed_loss_grad_vec).t()
                    #     # pdb.set_trace()
                    #     # print(' ',mask_val_unweighed_loss_grad_vec.max(),' ',mask_train_unweighed_loss_grad_vec.abs().max(),' ',implicit_gradient.abs().max())
                    #     # print(implicit_gradient[:5000].max(),' ',implicit_gradient[:5000].min())
                    #     # print(self._network.convnet.data_mask.data.max(),' ',self._network.convnet.data_mask.data.min(),' ',self._network.convnet.data_mask.data.sum()/5000)
                    #     # print('\n')
                    #     del mask_val_unweighed_loss_grad_vec
                    #     del mask_train_unweighed_loss_grad_vec
                    # assert self._network.convnet.data_mask.grad.abs().sum()==0
                    append_grad_to_vec(implicit_gradient, self._network)
                    optimizer.step()
                    # acc1, acc5 = self.accuracy(output, val_targets, topk=(1, 5))
                    losses_val += loss.item()
                    # if self._cur_task > 0:
                    #     pdb.set_trace()

            scheduler.step()
            if epoch%5==0:
                train_acc = self._compute_accuracy(self._network, train_loader)
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.3f}, Test_accy {:.3f}'.format(
                    self._cur_task, epoch, epochs, losses/len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}'.format(
                    self._cur_task, epoch, epochs, losses/len(train_loader))
            logging.info(info)

    def _run1(self, train_loader, val_loader, test_loader, optimizer, scheduler):
        lr2 = 1 # same with learning rate
        lambd = 0.1
        run_epochs = epochs
        self.change_vars("weight_mask")
        self.change_vars("weight")
        enabled, mask_enabled = set(), set()
        # qkv+mlp+proj:84934656 qkv+mlp: 77856768; qkv: 21233664
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        logging.info('Parameters to be updated: {}'.format(enabled))
        for epoch in range(1, run_epochs+1):
            self._network.train()
            losses, losses_val = 0., 0.
            correct, total = 0, 0
            losses = 0.
            # for i, (_, inputs, targets) in enumerate(train_loader):
            for i, (train_data_batch, val_data_batch) in enumerate(zip(train_loader, val_loader)):
                train_images, train_targets = train_data_batch[1].to(self._device), train_data_batch[2].to(self._device)
                val_images, val_targets = val_data_batch[1].to(self._device), val_data_batch[2].to(self._device)
                train_images_index, val_images_index = train_data_batch[0], val_data_batch[0]
                # 更新weight，用训练集。只打开weight
                # self.switch_mode(mode="finetune")
                # 当前batch内样本的mask
                output = self._network(train_images)['logits']
                weight = self._network.convnet.data_mask[train_images_index] # 不映射weight
                weight = torch.sigmoid(weight)
                loss = F.cross_entropy(output, train_targets)
                # loss = self.calculate_weighted_CE(output, train_targets)+loss_kd
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses += loss.item()
                _, preds = torch.max(output, dim=1)
                correct += preds.eq(train_targets.expand_as(preds)).cpu().sum()
                total += len(train_targets)
                ind = torch.where(preds == train_targets)[0] # 当前batch内预测正确的样本下标

            scheduler.step()
            if epoch%5==0:
                train_acc = self._compute_accuracy(self._network, train_loader)
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.3f}, Test_accy {:.3f}'.format(
                    self._cur_task, epoch, epochs, losses/len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}'.format(
                    self._cur_task, epoch, epochs, losses/len(train_loader))
            logging.info(info)

    def _stage1_training(self, train_loader, val_loader, test_loader):
        '''
        if self._cur_task == 0:
            loaded_dict = torch.load('./dict_0.pkl')
            self._network.load_state_dict(loaded_dict['model_state_dict'])
            self._network.to(self._device)
            return
        '''
        
        base_params = self._network.convnet.parameters()
        base_fc_params = [p for p in self._network.fc.parameters() if p.requires_grad==True]
        print('length of base_fc_params : ',len(base_fc_params))
        head_scale = 1. if 'moco' in self.log_path else 1.
        # lrate*self.bcb_lrscale lrate*head_scale
        base_params = {'params': base_params, 'lr': 0.001, 'weight_decay': weight_decay}
        base_fc_params = {'params': base_fc_params, 'lr': 0.001, 'weight_decay': weight_decay}
        network_params = [base_params, base_fc_params]
        optimizer = optim.SGD(network_params, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
        
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        
        if self.upper_bound:
            self._run_ori(train_loader, val_loader, test_loader, optimizer, scheduler)
        else:
            self._run_ori(train_loader, val_loader, test_loader, optimizer, scheduler)

    def calculate_class_means(self, model, data_loader):
        # calculate class means in train loader
        # class_means = torch.zeros(self._total_classes, self._network.feature_dim)
        print('Calculating class means...')
        model.eval()
        total = [0]*self._total_classes
        all_features = [[] for i in range(self._total_classes)]
        for i, (_, inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            with torch.no_grad():
                image_features = model(inputs)['features']
            vectors = image_features.clone().detach().cpu().numpy()
            for j in range(len(targets)):
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + 1e-8)).T
                total[targets[j]] += 1
                all_features[targets[j]].append(torch.tensor(vectors[j]))
        class_means = []
        for i in range(self._total_classes):
            features = torch.stack(all_features[i], dim=0).sum(dim=0)
            class_means.append(features / total[i])
        class_means = F.normalize(torch.stack(class_means, dim=0), p=2, dim=-1)
        return class_means
    
    def calculate_sparsity(self, model):
        all, count = 0, 0
        for i, v in model.named_modules():
            if hasattr(v, "adj"):
                attn = getattr(v, "adj")
                if not isinstance(attn, int):
                    param = attn.data
                    all += param.nelement()
                    count += torch.sum(param == 0)
                else:
                    return 0, 0, 0
        return count, all, np.round((100*count/all).cpu().numpy(), decimals=2)

    def switch_mode(self, mode='prune'):
        # print(f"#################### Pruning network ####################")
        # print(f"===>>  gradient for weights: None  | training importance scores only")
        assert mode in ["prune", "finetune", "bilevel", "prune_data"]
        if mode == 'finetune':
            self.change_vars("weight_mask", grad=False)
            self.change_vars("weight")
            self._network.convnet.data_mask.requires_grad_(False)
        elif mode == 'bilevel':
            self.change_vars("weight_mask")
            self.change_vars("weight")
            # self._network.convnet.data_mask.requires_grad_(True)
            self._network.convnet.data_mask.requires_grad_(False)
        elif mode == 'prune':
            self.change_vars("weight_mask")
            self.change_vars("weight", grad=False)
            # self._network.convnet.data_mask.requires_grad_(True)
            self._network.convnet.data_mask.requires_grad_(False)
        else:
            self.change_vars("weight_mask", grad=False)
            self.change_vars("weight", grad=False)
            self._network.convnet.data_mask.requires_grad_(True)
        
    def change_vars(self, var_name, freeze_bn=False, grad=True):
        """
        freeze vars. If freeze_bn then only freeze batch_norm params.
        """
        # var_name = qkv.weight, qkv.weight_mask, mlp.fc1.weight, mlp.fc1.weight_mask, mlp.fc2.weight, mlp.fc2.weight_mask
        # assert var_name in ["weight", "bias", "weight_mask"]
        for name, param in self._network.named_parameters():
            if ("mask" in var_name):
                if var_name in name:
                    param.requires_grad_(grad)
            else:
                if self.model == 'vit':
                    # if (("mlp" in name)or("qkv" in name)or("attn.proj" in name))and("mask" not in name):
                    if (("mlp" in name)or("qkv" in name))and("mask" not in name):
                        param.requires_grad_(grad)
                elif 'resnet' in self.model:
                    if ("mask" not in name):
                        param.requires_grad_(grad)
                # if ("fc.weight" == name)or("fc.bias" == name):
                #     param.requires_grad_(grad)

    def calculate_weighted_CE(self, output, targets, weight=None):
        t = 0
        if weight == None:
            weight = torch.ones(len(targets)).to(targets.device)
        # 先类内平均再所有样本平均，考虑了weight映射成0,1的情况
        unique_targets = torch.unique(targets)
        ind_w = torch.where(weight != 0)[0].tolist() # 权重不为0的weight下标
        # for i in unique_targets:
        #     ind_tar = torch.where(targets == i)[0].tolist()
        #     ind = set(ind_tar)&set(ind_w)
        #     tt = 0
        #     if len(ind) != 0:
        #         for j in ind:
        #             tt = tt + weight[j]*F.cross_entropy(output[j], targets[j])
        #         tt = tt / len(ind)
        #     t = t+tt
        # t = t / len(unique_targets)
        # 计算加权平均
        for i in range(output.shape[0]):
            t = t + weight[i]*F.cross_entropy(output[i:i+1], targets[i:i+1])
        # t /= output.shape[0]
        t /= weight[weight!=0].shape[0]
        return t
    
    def crop_and_save(self, file_name, ratio, quality, save_dir):
        # file_name: 样本路径表。下面算裁剪前和裁剪后的图像大小
        # train_trsf = [transforms.RandomResizedCrop(224),]
        new_file_name = []
        ori_space, space, file_num = 0, 0, 0
        for img_ori_path in file_name:
            if 'data_buffer' not in img_ori_path:
                part = img_ori_path.split('/')
                new_name = os.path.join(save_dir,part[-2]+'_'+part[-1])
                
                img_size = Image.open(img_ori_path).size
                width, height = img_size[0]*ratio, img_size[1]*ratio
                train_trsf = [transforms.CenterCrop((height, width)),]
                # train_trsf = [transforms.CenterCrop((375, 375)),]
                trsf = transforms.Compose([*train_trsf])
                img =  trsf(pil_loader(img_ori_path))
                # train_trsf = [
                #     transforms.CenterCrop((height, width)),
                #     # transforms.RandomResizedCrop((height, width)),
                #     # transforms.Resize((400, 400)),
                # ]
                # trsf = transforms.Compose([*train_trsf])
                # img = trsf(img_ori)
                # img_ori = pil_loader(img_ori_path)
                # img = img_ori
                # if img_size[0] > 400 and img_size[1] > 400:
                #     img = trsf(img_ori)
                    
                # 用PIL自带的裁剪
                # img_ori = Image.open(img_ori_path)
                # img_size = img_ori.size
                # width, height = img_size[0]*ratio, img_size[1]*ratio
                # begin, end = img_size[0]*(1-ratio)/2, img_size[1]*(1-ratio)/2
                # img = img_ori.crop((begin, end, begin+width, end+height))
                
                img.save(new_name, quality=quality)
                space += os.path.getsize(new_name)
                ori_space += os.path.getsize(img_ori_path)
                file_num += 1
                new_file_name.append(new_name)
            else:
                new_file_name.append(img_ori_path)
        self.saved_sample_space.append(space/file_num)
        logging.info('curr task saved img space/ori img space | all avg saved img space:{}/{} | {}'.format(space, ori_space, sum(self.saved_sample_space)/len(self.saved_sample_space)))
        return np.array(new_file_name)
    
