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


# JMOCL
class SLCA(BaseLearner):
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

        self.seed = args['seed']
        self.task_sizes = []
        
        self.args = args
        self.model = 'vit'
        self.cnn_acc, self.nme_acc = [], []
        self.remain_params = []
        self.gamma = self.args['gamma']
        self.saved_sample_space = [] # only for imagenetr
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
            # self.mask_ratio = self.mask_ratio-(self._cur_task//2+1)/100
            self.mask_ratio = 1-self.mask_ratio_per_task*(self._cur_task+1)
            # self.mask_ratio = 0.4+0.03*(self._cur_task+1)
            # self.mask_ratio = 0.99
            # if self._cur_task == 9:
            #     self.mask_ratio = 0.7
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        enabled, mask_enabled = set(), set()
        if self.model == 'vit':
            # qkv+mlp: 77856768; qkv: 21233664
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
                
                self.switch_mode(mode="finetune")
                output = self._network(train_images)['logits']
                weight = self._network.convnet.data_mask[train_images_index]
                weight = torch.sigmoid(weight)
                loss_kd = 0
                loss = self.calculate_weighted_CE(output, train_targets, weight.abs())+loss_kd
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses += loss.item()
                _, preds = torch.max(output, dim=1)
                correct += preds.eq(train_targets.expand_as(preds)).cpu().sum()
                total += len(train_targets)
                ind = torch.where(preds == train_targets)[0]

                T_epoch, w = 0, 1
                if i % (self.args['update_step']) == 0 and epoch >= T_epoch:
                    self.switch_mode(mode="bilevel")
                    optimizer.zero_grad()
                    output = self._network(val_images)['logits']
                    loss_kd = 0
                    loss = F.cross_entropy(output, val_targets)+loss_kd
                    loss.backward()
                    all_val_unweighted_loss_grad_vec = grad2vec(self._network)
                    if self._network.convnet.data_mask.grad is None:
                        tt = torch.zeros(self._network.convnet.data_mask.shape[0]).to(self._device)
                        all_val_unweighted_loss_grad_vec = torch.cat((tt, all_val_unweighted_loss_grad_vec), dim=0)
                    
                    self.switch_mode(mode="prune")
                    optimizer.zero_grad()
                    output = self._network(train_images)["logits"]
                    weight = self._network.convnet.data_mask[train_images_index]
                    weight = torch.sigmoid(weight)
                    loss_kd = 0
                    loss_mask = self.calculate_weighted_CE(output, train_targets, weight.abs())+loss_kd
                    loss_mask.backward()
                    
                    mask_train_weighted_loss_grad_vec = grad2vec(self._network)
                    implicit_gradient = -lr2 * (1/self.gamma) * mask_train_weighted_loss_grad_vec * all_val_unweighted_loss_grad_vec
                    del mask_train_weighted_loss_grad_vec

                    output = self._network(train_images)['logits']
                    loss_kd = 0
                    loss = F.cross_entropy(output, train_targets)+loss_kd
                    optimizer.zero_grad()
                    loss.backward()
                    ori_grad = grad2vec(self._network)
                    loss_ori = loss.item()
                    for j in range(train_images.shape[0]):
                        with torch.no_grad():
                            loss1 = F.cross_entropy(output[j:j+1], train_targets[j:j+1]).item() + loss_kd
                        mask_train_unweighed_loss_grad_vec = loss1/(train_images.shape[0]*loss_ori) * ori_grad
                        implicit_gradient[train_images_index[j]] = -w*50*all_val_unweighted_loss_grad_vec@((1/self.gamma)*mask_train_unweighed_loss_grad_vec).t()
                    del all_val_unweighted_loss_grad_vec
                    del mask_train_unweighed_loss_grad_vec
                    append_grad_to_vec(implicit_gradient, self._network)
                    optimizer.step()
                    losses_val += loss.item()

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
                    if (("mlp" in name)or("qkv" in name))and("mask" not in name):
                        param.requires_grad_(grad)
                elif 'resnet' in self.model:
                    if ("mask" not in name):
                        param.requires_grad_(grad)

    def calculate_weighted_CE(self, output, targets, weight=None):
        t = 0
        if weight == None:
            weight = torch.ones(len(targets)).to(targets.device)

        for i in range(output.shape[0]):
            t = t + weight[i]*F.cross_entropy(output[i:i+1], targets[i:i+1])
        t /= weight[weight!=0].shape[0]
        return t
    
    def crop_and_save(self, file_name, ratio, quality, save_dir):
        new_file_name = []
        ori_space, space, file_num = 0, 0, 0
        for img_ori_path in file_name:
            if 'data_buffer' not in img_ori_path:
                part = img_ori_path.split('/')
                new_name = os.path.join(save_dir,part[-2]+'_'+part[-1])
                
                img_size = Image.open(img_ori_path).size
                width, height = img_size[0]*ratio, img_size[1]*ratio
                train_trsf = [transforms.CenterCrop((height, width)),]
                trsf = transforms.Compose([*train_trsf])
                img =  trsf(pil_loader(img_ori_path))
                
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
