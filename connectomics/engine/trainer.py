import os, sys, glob 
import time, itertools
import GPUtil
from yacs.config import CfgNode

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .solver import *
from connectomics.model import *
from connectomics.data.augmentation import build_train_augmentor, TestAugmentor
from connectomics.data.dataset import build_dataloader, get_dataset
from connectomics.data.utils import build_blending_matrix, writeh5

class Trainer(object):
    r"""Trainer

    Args:
        cfg (yacs.config.CfgNode): YACS configuration options.
        device (torch.device): by default all training and inference are conducted on GPUs.
        mode (str): running mode of the trainer (``'train'`` or ``'test'``).
        checkpoint (optional): the checkpoint file to be loaded (default: `None`)
    """
    def __init__(self, cfg, device, mode, checkpoint=None):
        self.cfg = cfg
        self.device = device
        self.output_dir = cfg.DATASET.OUTPUT_PATH
        self.mode = mode

        self.model = build_model(self.cfg, self.device)
        self.optimizer = build_optimizer(self.cfg, self.model)
        self.lr_scheduler = build_lr_scheduler(self.cfg, self.optimizer)
        self.start_iter = self.cfg.MODEL.PRE_MODEL_ITER
        if checkpoint is not None:
            self.update_checkpoint(checkpoint)

        if self.mode == 'train':
            self.augmentor = build_train_augmentor(self.cfg)
            self.monitor = build_monitor(self.cfg)
            self.criterion = build_criterion(self.cfg, self.device)
            # add config details to tensorboard
            self.monitor.load_config(self.cfg)
        else:
            self.augmentor = None

        if cfg.DATASET.DO_CHUNK_TITLE == 0:
            self.dataloader = build_dataloader(self.cfg, self.augmentor, self.mode)
            self.dataloader = iter(self.dataloader)
        else:
            self.dataset = None
            self.dataloader = None

        self.total_iter_nums = self.cfg.SOLVER.ITERATION_TOTAL - self.start_iter
        self.inference_output_name = self.cfg.INFERENCE.OUTPUT_NAME


    def train(self):
        r"""Training function.
        """
        # setup
        self.model.train()
        self.monitor.reset()
        self.optimizer.zero_grad()

        for iteration in range(self.total_iter_nums):
            iter_total = self.start_iter + iteration
            start = time.perf_counter()

            # load data
            batch = next(self.dataloader)
            _, volume, target, weight = batch
            time1 = time.perf_counter()

            # prediction
            volume = torch.from_numpy(volume).to(self.device, dtype=torch.float)
            pred = self.model(volume)
           
            loss = self.criterion.eval(pred, target, weight)

            # compute gradient
            loss.backward()
            if (iteration+1) % self.cfg.SOLVER.ITERATION_STEP == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            # logging and update record
            do_vis = self.monitor.update(self.lr_scheduler, iter_total, loss, self.optimizer.param_groups[0]['lr']) 
            if do_vis:
                self.monitor.visualize(volume, target, pred, iter_total)
                # Display GPU stats using the GPUtil package.
                GPUtil.showUtilization(all=True)

            # Save model
            if (iter_total+1) % self.cfg.SOLVER.ITERATION_SAVE == 0:
                self.save_checkpoint(iter_total)

            # update learning rate
            self.lr_scheduler.step(loss) if self.cfg.SOLVER.LR_SCHEDULER_NAME == 'ReduceLROnPlateau' else self.lr_scheduler.step()

            end = time.perf_counter()
            print('[Iteration %05d] Data time: %.5f, Iter time:  %.5f' % (iter_total, time1 - start, end - start))

            # Release some GPU memory and ensure same GPU usage in the consecutive iterations according to 
            # https://discuss.pytorch.org/t/gpu-memory-consumption-increases-while-training/2770
            del loss, pred

    def test(self):
        r"""Inference function.
        """
        if self.cfg.INFERENCE.DO_EVAL:
            self.model.eval()
        else:
            self.model.train()

        ww = build_blending_matrix(self.cfg.MODEL.OUTPUT_SIZE, self.cfg.INFERENCE.BLENDING)
        if self.cfg.INFERENCE.MODEL_OUTPUT_ID[0] is None:
            NUM_OUT = self.cfg.MODEL.OUT_PLANES
        else:
            NUM_OUT = len(self.cfg.INFERENCE.MODEL_OUTPUT_ID)
        pad_size = self.cfg.DATASET.PAD_SIZE
        if len(self.cfg.DATASET.PAD_SIZE)==3:
            pad_size = [self.cfg.DATASET.PAD_SIZE[0],self.cfg.DATASET.PAD_SIZE[0],
                        self.cfg.DATASET.PAD_SIZE[1],self.cfg.DATASET.PAD_SIZE[1],
                        self.cfg.DATASET.PAD_SIZE[2],self.cfg.DATASET.PAD_SIZE[2]]
        
        if ("super" in self.cfg.MODEL.ARCHITECTURE):
            output_size = np.array(self.dataloader._dataset.volume_size)*np.array(self.cfg.DATASET.SCALE_FACTOR).tolist()
            result = [np.stack([np.zeros(x, dtype=np.float32) for _ in range(NUM_OUT)]) for x in output_size]
            weight = [np.zeros(x, dtype=np.float32) for x in output_size]
        else:
            result = [np.stack([np.zeros(x, dtype=np.float32) for _ in range(NUM_OUT)]) for x in self.dataloader._dataset.volume_size]
            weight = [np.zeros(x, dtype=np.float32) for x in self.dataloader._dataset.volume_size]

        # build test-time augmentor and update output filename
        test_augmentor = TestAugmentor(self.cfg.INFERENCE.AUG_MODE, 
                                       self.cfg.INFERENCE.AUG_NUM)
        self.inference_output_name = test_augmentor.update_name(self.inference_output_name)

        start = time.time()
        sz = tuple([NUM_OUT] + list(self.cfg.MODEL.OUTPUT_SIZE))
        total_num_vols = len(self.dataloader) * self.cfg.INFERENCE.SAMPLES_PER_BATCH
        print("Total number of volumes: ", total_num_vols)

        volume_id = 0
        with torch.no_grad():
            for _, (pos, volume) in enumerate(self.dataloader):
                volume_id += self.cfg.INFERENCE.SAMPLES_PER_BATCH
                print('progress: %d/%d' % (volume_id, total_num_vols))

                # for gpu computing
                volume = torch.from_numpy(volume).to(self.device)
                if not self.cfg.INFERENCE.DO_3D:
                    volume = volume.squeeze(1)

                # forward pass
                output = test_augmentor(self.model, volume)
                # select channel, self.cfg.INFERENCE.MODEL_OUTPUT_ID is a list [None]
                if self.cfg.INFERENCE.MODEL_OUTPUT_ID[0] is not None: 
                    ndim = output.ndim
                    output = output[:, self.cfg.INFERENCE.MODEL_OUTPUT_ID[0]]
                    if ndim - output.ndim == 1:
                        output = output[:,None,:]
                if not "super" in self.cfg.MODEL.ARCHITECTURE:
                    for idx in range(output.shape[0]):
                        st = pos[idx]
                        if result[st[0]].ndim - output[idx].ndim == 1:
                            result[st[0]][:, st[1]:st[1]+sz[1], st[2]:st[2]+sz[2], \
                                          st[3]:st[3]+sz[3]] += output[idx][:,None,:] * ww[None,:]
                        else:
                            result[st[0]][:, st[1]:st[1]+sz[1], st[2]:st[2]+sz[2], \
                                        st[3]:st[3]+sz[3]] += output[idx] * ww[None,:]
                        weight[st[0]][st[1]:st[1]+sz[1], st[2]:st[2]+sz[2], \
                        st[3]:st[3]+sz[3]] += ww
                else:
                    for idx in range(output.shape[0]):
                        st = pos[idx]
                        st = (np.array(st)*np.array([1]+self.cfg.DATASET.SCALE_FACTOR)).tolist()
                        result[st[0]][:, st[1]:st[1]+sz[1], st[2]:st[2]+sz[2], \
                        st[3]:st[3]+sz[3]] += output[idx] * np.expand_dims(ww, axis=0)
                        weight[st[0]][st[1]:st[1]+sz[1], st[2]:st[2]+sz[2], \
                        st[3]:st[3]+sz[3]] += ww

        end = time.time()
        print("Prediction time:", (end-start))

        for vol_id in range(len(result)):
            if result[vol_id].ndim > weight[vol_id].ndim:
                weight[vol_id] = np.expand_dims(weight[vol_id], axis=0)
            # For segmentation masks, use uint16
            result[vol_id] = (result[vol_id]/weight[vol_id]*255).astype(np.uint8)
            sz = result[vol_id].shape
            result[vol_id] = result[vol_id][:,
                        pad_size[0]:sz[1]-pad_size[1],
                        pad_size[2]:sz[2]-pad_size[3],
                        pad_size[4]:sz[3]-pad_size[5]]

        if self.output_dir is None:
            return result
        else:
            print('Saving as h5...')
            writeh5(os.path.join(self.output_dir, self.inference_output_name), result,
                    ['vol%d'%(x) for x in range(len(result))])
            print('Inference is done!')

    # -----------------------------------------------------------------------------
    # Misc functions
    # -----------------------------------------------------------------------------
    def save_checkpoint(self, iteration):
        state = {'iteration': iteration + 1,
                 'state_dict': self.model.module.state_dict(), # Saving torch.nn.DataParallel Models
                 'optimizer': self.optimizer.state_dict(),
                 'lr_scheduler': self.lr_scheduler.state_dict()}
                 
        # Saves checkpoint to experiment directory
        filename = 'checkpoint_%05d.pth.tar' % (iteration + 1)
        filename = os.path.join(self.output_dir, filename)
        torch.save(state, filename)

    def update_checkpoint(self, checkpoint):
        # load pre-trained model
        print('Load pretrained checkpoint: ', checkpoint)
        checkpoint = torch.load(checkpoint)
        print('checkpoints: ', checkpoint.keys())
        
        # update model weights
        if 'state_dict' in checkpoint.keys():
            pretrained_dict = checkpoint['state_dict']
            model_dict = self.model.module.state_dict() # nn.DataParallel
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict 
            model_dict.update(pretrained_dict)    
            # 3. load the new state dict
            self.model.module.load_state_dict(model_dict) # nn.DataParallel   

        if not self.cfg.SOLVER.ITERATION_RESTART:
            # update optimizer
            if 'optimizer' in checkpoint.keys():
                self.optimizer.load_state_dict(checkpoint['optimizer'])

            # update lr scheduler
            if 'lr_scheduler' in checkpoint.keys():
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            # load iteration
            if 'iteration' in checkpoint.keys():
                self.start_iter = checkpoint['iteration']

    def run_chunk(self, mode):
        self.dataset = get_dataset(self.cfg, self.augmentor, mode)
        if mode == 'train':
            num_chunk = self.total_iter_nums // self.cfg.DATASET.DATA_CHUNK_ITER
            self.total_iter_nums = self.cfg.DATASET.DATA_CHUNK_ITER
            for chunk in range(num_chunk):
                self.dataset.updatechunk()
                self.dataloader = build_dataloader(self.cfg, self.augmentor, mode, 
                                                   dataset=self.dataset.dataset)
                self.dataloader = iter(self.dataloader)
                print('start train', chunk)
                self.train()
                print('finished train', chunk)
                self.start_iter += self.cfg.DATASET.DATA_CHUNK_ITER
                del self.dataloader
        else:
            num_chunk = len(self.dataset.chunk_num_ind)
            for chunk in range(num_chunk):
                self.dataset.updatechunk(do_load=False)
                self.inference_output_name = self.cfg.INFERENCE.OUTPUT_NAME + self.dataset.get_coord_name() + '.h5'
                if not os.path.exists(os.path.join(self.output_dir, self.inference_output_name)):
                    self.dataset.loadchunk()
                    self.dataloader = build_dataloader(self.cfg, self.augmentor, mode, 
                                                       dataset=self.dataset.dataset)
                    self.dataloader = iter(self.dataloader)
                    self.test()
