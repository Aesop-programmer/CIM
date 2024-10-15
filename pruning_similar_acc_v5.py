
from datasets import load_dataset
import argparse
import torch
torch.manual_seed(91123)
parser = argparse.ArgumentParser(description='Pruning')
parser.add_argument('--pruning_type', default='l1', type=str, help='pruning type', choices=['random', 'taylor', 'l1','hessian','similar'])
parser.add_argument('--iter',default=1, type=int, help='pruning iteration')
parser.add_argument('--model',type=str,help="model architecture")
parser.add_argument('--model_path', type=str, help='pruning model path')
parser.add_argument('--thr', type=float, help='thr')
parser.add_argument('--reuse', help='resue',action='store_true')
args = parser.parse_args()


# load cifar10 (only small portion for demonstration purposes) 
train_ds, test_ds = load_dataset('cifar100', split=['train[:]', 'test[:]'])
# split up training into training + validation
splits = train_ds.train_test_split(test_size=0.1)
train_ds = splits['train']
val_ds = splits['test']

# %%
id2label = {id:label for id, label in enumerate(train_ds.features['fine_label'].names)}
label2id = {label:id for id, label in id2label.items()}



# %%
from transformers import ViTImageProcessor
from transformers import AutoImageProcessor, DeiTForImageClassification
if "vit" in args.model:
    processor = ViTImageProcessor.from_pretrained(args.model,size=224)
elif "deit" in args.model:
    processor = AutoImageProcessor.from_pretrained(args.model,size=224)
else: raise NotImplementedError
image_mean = processor.image_mean
image_std = processor.image_std
size = processor.size["height"]

# %%
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor)

normalize = Normalize(mean=image_mean, std=image_std)
_train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

_val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]
    return examples

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]
    return examples

# %%
# Set the transforms
train_ds.set_transform(train_transforms)
val_ds.set_transform(val_transforms)
test_ds.set_transform(val_transforms)


# %%
from torch.utils.data import DataLoader
import torch

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["fine_label"] for example in examples])
    return {"pixel_values": pixel_values, "fine_labels": labels}

train_batch_size = 32
eval_batch_size = 32

train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=collate_fn, batch_size=train_batch_size,num_workers=10)
val_dataloader = DataLoader(val_ds, collate_fn=collate_fn, batch_size=eval_batch_size,num_workers=10)
test_dataloader = DataLoader(test_ds, collate_fn=collate_fn, batch_size=eval_batch_size,num_workers=10)



# %%
import pytorch_lightning as pl
from transformers import ViTForImageClassification, AdamW
import torch.nn as nn

class ViTLightningModule(pl.LightningModule):
    def __init__(self, num_labels=100):
        super(ViTLightningModule, self).__init__()
        if "vit" in args.model: 
            self.vit = ViTForImageClassification.from_pretrained(args.model,
                                                              num_labels=100,
                                                              id2label=id2label,
                                                              label2id=label2id)
        elif "deit" in args.model:
            self.vit = DeiTForImageClassification.from_pretrained(args.model,
                                                              num_labels=100,
                                                              id2label=id2label,
                                                              label2id=label2id)
        else: raise NotImplementedError
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits
        
    def common_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['fine_labels']
        logits = self(pixel_values)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        predictions = logits.argmax(-1)
        correct = (predictions == labels).sum().item()
        accuracy = correct/pixel_values.shape[0]

        return loss, accuracy
      
    def training_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        self.log("training_accuracy", accuracy)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss, on_epoch=True)
        self.log("validation_accuracy", accuracy, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     
        self.log("testing_loss",loss,on_epoch=True)     
        self.log("testing_accuracy",accuracy,on_epoch=True)
        return loss

    def configure_optimizers(self):
        # We could make the optimizer more fancy by adding a scheduler and specifying which parameters do
        # not require weight_decay but just using AdamW out-of-the-box works fine
        return AdamW(self.parameters(), lr=5e-5)

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader

    def test_dataloader(self):
        return test_dataloader

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

# for early stopping, see https://pytorch-lightning.readthedocs.io/en/1.0.0/early_stopping.html?highlight=early%20stopping
early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=10,
    strict=False,
    verbose=False,
    mode='min'
)

# %%
from transformers import ViTConfig, ViTForImageClassification
import torch_pruning as tp

from transformers.models.vit.modeling_vit import ViTSelfAttention, ViTSelfOutput
from transformers.models.deit.modeling_deit import DeiTSelfAttention


# %%
from torchvision.models.vision_transformer import VisionTransformer

model =ViTLightningModule.load_from_checkpoint(args.model_path).eval().to("cuda")

example_inputs = torch.randn(1,3,224,224).to("cuda")

if "vit" in args.model: 
    num_heads = {}
    ignored_layers = [model.vit.classifier]
    # All heads should be pruned simultaneously, so we group channels by head.
    for m in model.modules():
        if isinstance(m, ViTSelfAttention):
            num_heads[m.query] = m.num_attention_heads
            num_heads[m.key] = m.num_attention_heads
            num_heads[m.value] = m.num_attention_heads
elif "deit" in args.model:
    num_heads = {}
    ignored_layers = [model.vit.classifier]
    for m in model.modules():
        if isinstance(m, DeiTSelfAttention):
            num_heads[m.query] = m.num_attention_heads
            num_heads[m.key] = m.num_attention_heads
            num_heads[m.value] = m.num_attention_heads
else:raise NotImplementedError

if args.pruning_type == 'random':
    imp = tp.importance.RandomImportance()
elif args.pruning_type == 'taylor':
    imp = tp.importance.GroupTaylorImportance()
elif args.pruning_type == 'l1':
    imp = tp.importance.MagnitudeImportance(p=1)
elif args.pruning_type == 'hessian':
    imp = tp.importance.GroupHessianImportance()
elif args.pruning_type == 'similar':
    imp = tp.importance.SimilarImportance()
else: raise NotImplementedError

import glob

def forward_pre_hook(self, inputs):
    # Allows for examination and modification of the input before the forward pass.
    # Note that inputs are always wrapped in a tuple.
    if args.reuse:
        input = inputs[0]
        if len(input.shape) == 3:
            input = torch.flatten(input,start_dim=0,end_dim=1)
        input_mean = torch.mean(input,0)
        N = self.get_buffer('N').to('cuda')
        input_average = self.get_buffer('input_average').to('cuda')
        input_average = input_average*N/(N+torch.tensor(1.)) + input_mean/(N+torch.tensor(1.))
        N = N + torch.tensor(1.)
        
        
        input_buffer = self.get_buffer('input_buffer')
        input_buffer = torch.cat([input_buffer,input.clone().detach().cpu()])
        
        self.input_buffer = input_buffer
        self.N = N
        self.input_average = input_average
    
    return inputs


from tensordict import TensorDict
import os

import tempfile
import random
import numpy as np
def init_hook_input(model):
    # global file_name
    for m in model.modules():
        if isinstance(m,nn.Linear):
            m.register_buffer('N',torch.tensor(0.),persistent=False)
            input_dim = m.weight.shape[1]
            input_average = torch.zeros(input_dim,dtype=torch.float32)
            m.register_buffer('input_average',input_average,persistent=False)
            # fp = tempfile.NamedTemporaryFile(prefix=str(input_layer_activation_idex),suffix=".pt",delete=False)
            m.register_buffer('input_buffer',torch.tensor([]),persistent=False)
            
            
def relase_input_buffer(model):
    for m in model.modules():
        if isinstance(m,nn.Linear):
            del m.N
            del m.input_average
            del m.input_buffer
            
pruner = tp.pruner.MetaPruner(
                model, 
                example_inputs,
                iterative_steps=args.iter,
                global_pruning=False, # If False, a uniform pruning ratio will be assigned to different layers.
                importance=imp, # importance criterion for parameter selection
                pruning_ratio=0.9, # target pruning ratio
                # unwrapped_parameters=unwrapped_parameters,
                ignored_layers=ignored_layers,
                # output_transform=lambda out: out.logits.sum(),
                # num_heads=num_heads,
                num_heads = num_heads,
                prune_head_dims=False,
                prune_num_heads=True,
                head_pruning_ratio=(1.0/3.0), # disabled when prune_num_heads=False
                round_to=512,
)
        
@torch.no_grad()
def calculate_relation(module):
    if isinstance(module,nn.Linear):
        input_dim = module.weight.shape[1] ## number of column vector
        output_dim = module.weight.shape[0] ## column vector dim
        similar_matrix = torch.zeros([output_dim,input_dim],dtype=torch.float32).to('cuda')
        similar_temp = torch.zeros([input_dim,input_dim],dtype=torch.float32).to('cuda')
        similar_matrix_tra = torch.transpose(similar_matrix,0,1)
        similar_temp_tra = torch.transpose(similar_temp,0,1)
        for i in range(input_dim):
            temp = torch.tensor(0,dtype=torch.float32).to('cuda')
            for j in range(i+1,input_dim):
                # does not inner product itself
                if i == j:
                    pass
                else:
                    similar_temp_tra[i][j] = torch.matmul(torch.transpose((module.weight.data),0,1)[i],torch.transpose(module.weight.data,0,1)[j])/((torch.norm(torch.transpose((module.weight.data),0,1)[i])*torch.norm(torch.transpose((module.weight.data),0,1)[j])))
            for k in range(input_dim):
                if k < i :
                    temp += torch.abs(similar_temp_tra[k][i])
                elif i > k:
                    tmep += torch.abs(similar_temp_tra[i][k])
            temp = temp / (output_dim - 1)
            # temp = 1 / (temp + 1e-12)
            similar_matrix_tra[i] = torch.add(similar_matrix_tra[i],temp)
        similar_matrix = torch.transpose(similar_matrix_tra,0,1)
        module.register_buffer('similar_table',similar_temp_tra)
        module.register_buffer('similar inverse',similar_matrix,persistent=False)
    elif isinstance(module,nn.modules.conv._ConvNd):
        input_dim = module.weight.shape[1]
        output_dim =module.weight.shape[0]
        similar_matrix = torch.zeros([output_dim,input_dim,module.weight.shape[2],module.weight.shape[3]],dtype=torch.float32).to('cuda')
        if (module.weight.shape != similar_matrix.shape):
            raise Exception('conv similar matrix dim error')
        for i in range(output_dim):
            temp = torch.tensor(0,dtype=torch.float32).to('cuda')
            for j in range(output_dim):
                if i == j:
                    pass
                else:
                    temp += torch.sum(torch.mul(module.weight[i],module.weight[j])/(torch.norm(module.weight[i])*torch.norm(module.weight[j])))
            temp = temp / (output_dim - 1)
            # temp = 1 / (temp + 1e-12)
            similar_matrix[i] = torch.add(similar_matrix[i],temp)
        module.register_buffer('similar inverse',similar_matrix,persistent=False)


def calculate_importance():
    if isinstance(imp, (tp.importance.GroupTaylorImportance, tp.importance.GroupHessianImportance)):
        model.to("cuda")
        model.zero_grad()
        if isinstance(imp, tp.importance.GroupHessianImportance):
            imp.zero_grad()
        print("Accumulating gradients for pruning...")
        for k, (batch) in enumerate(train_dataloader):
            if k>=10: break
            imgs = batch["pixel_values"]
            label = batch["fine_labels"]
            imgs = imgs.to("cuda")
            label = label.to("cuda")
            output = model(imgs)
            if isinstance(imp, tp.importance.GroupHessianImportance):
                loss = torch.nn.functional.cross_entropy(output, label, reduction='none')
                for l in loss:
                    model.to("cuda")
                    model.zero_grad()
                    l.backward(retain_graph=True)
                    imp.accumulate_grad(model)
            elif isinstance(imp, tp.importance.GroupTaylorImportance):
                loss = torch.nn.functional.cross_entropy(output, label)
                loss.backward()
    elif isinstance(imp, (tp.importance.SimilarImportance)):
        print('calculating similar matrix')
        target = [nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm]
        for m in model.modules():
            if isinstance(m, tuple(target)):
                calculate_relation(m)
        # ## statistics of input
        # model.to("cuda")
        # model.zero_grad()
        # model.eval()
        # init_hook_input(model)
        # with torch.no_grad():
        #     for k , (batch) in enumerate(train_dataloader):
        #         # if k>=2: break
        #         imgs = batch["pixel_values"]
        #         imgs = imgs.to("cuda")
        #         output = model(imgs)
        #         torch.cuda.empty_cache()

def store_input():
    ## statistics of input
    model.to("cuda")
    model.eval()
    init_hook_input(model)
    
    hooks = list()
    for m in model.modules():
        if isinstance(m,nn.Linear):
            hook = m.register_forward_pre_hook(forward_pre_hook)
            hooks.append(hook)
    
    with torch.no_grad():
        for k , (batch) in enumerate(train_dataloader):
            # if k>=2: break
            # print(k)
            if k >= 40: break
            imgs = batch["pixel_values"]
            imgs = imgs.to("cuda")
            output = model(imgs)
    torch.cuda.empty_cache()
    for hook in hooks:
        hook.remove()

from lightning.pytorch.loggers import TensorBoardLogger

import os
os.environ["threshold"] = str(args.thr)
os.environ["counter"] = "0"
os.environ["reuse"] = str(args.reuse)

if args.iter == 1:
    for i in range(args.iter):

        
      
        logger = TensorBoardLogger("~/nas/homes/aesop/pruning_once_reuse_v5", name= args.model+args.pruning_type+str(args.reuse)+"/step:"+str(i))
        trainer = Trainer(callbacks=[EarlyStopping(monitor='validation_loss')],logger=logger)
        print(i)
        if isinstance(imp, (tp.importance.GroupTaylorImportance, tp.importance.GroupHessianImportance,tp.importance.SimilarImportance)):
            calculate_importance()
            
        ### analyze input
        if args.reuse:
            print("reuse")
            store_input()
        
        pruner.step(interactive=False)
        
        # Modify the attention head size and all head size aftering pruning
        if "vit" in args.model:
            for m in model.modules():
                if isinstance(m, ViTSelfAttention):
                    m.num_attention_heads = pruner.num_heads[m.query]
                    m.attention_head_size = m.query.out_features // m.num_attention_heads
                    m.all_head_size = m.query.out_features
                    
        elif "deit" in args.model:
            for m in model.modules():
                if isinstance(m, DeiTSelfAttention):
                    m.num_attention_heads = pruner.num_heads[m.query]
                    m.attention_head_size = m.query.out_features // m.num_attention_heads
                    m.all_head_size = m.query.out_features
        else:raise NotImplementedError
        # base_macs, base_params = tp.utils.count_ops_and_params(model, torch.randn(1,3,224,224).to("cuda"))
        # print(base_macs/1e9, base_params/1e6)
        if args.reuse:
            relase_input_buffer(model)
        torch.set_float32_matmul_precision("highest")
        model.to("cuda")
        trainer.fit(model)
        trainer.test(model)
        
else:
    for i in range(args.iter):
         ### initializtion register
        ### analyze input
        
        
        
        logger = TensorBoardLogger("~/nas/homes/aesop/pruning_iter_reuse_v5_"+f'{args.iter}'+"_acc", name= args.model+args.pruning_type+str(args.reuse)+"/step:"+str(i))
        trainer = Trainer(callbacks=[EarlyStopping(monitor='validation_loss')],logger=logger)
        print(i)
        if isinstance(imp, (tp.importance.GroupTaylorImportance, tp.importance.GroupHessianImportance,tp.importance.SimilarImportance)):
            calculate_importance()
        if args.reuse:                          
            print("reuse")

            store_input()
        pruner.step(interactive=False)
        # Modify the attention head size and all head size aftering pruning
        if "vit" in args.model:
            for m in model.modules():
                if isinstance(m, ViTSelfAttention):
                    m.num_attention_heads = pruner.num_heads[m.query]
                    m.attention_head_size = m.query.out_features // m.num_attention_heads
                    m.all_head_size = m.query.out_features
                    
        elif "deit" in args.model:
            for m in model.modules():
                if isinstance(m, DeiTSelfAttention):
                    m.num_attention_heads = pruner.num_heads[m.query]
                    m.attention_head_size = m.query.out_features // m.num_attention_heads
                    m.all_head_size = m.query.out_features
        else:raise NotImplementedError
        # base_macs, base_params = tp.utils.count_ops_and_params(model, torch.randn(1,3,224,224).to("cuda"))
        # print(base_macs/1e9, base_params/1e6)
        if args.reuse:
            relase_input_buffer(model)
        torch.set_float32_matmul_precision("highest")
        model.to("cuda")
        trainer.fit(model)
        trainer.test(model)
print("method:",args.pruning_type)
print("threshold:",os.environ["threshold"])
print("counter:",os.environ["counter"])






