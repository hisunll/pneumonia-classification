import codecs
import os
import shutil
from PIL import Image
import csv
import torch
from torch.nn.modules import *
from functools import partial
import math
import time
"""all_file_dir='./Data/Data'
train_file_dir = './Data/Data/train'
test_file_dir='./Data/Data/test'
val_file_dir='./Data/Data/val'
class_list = [c for c in os.listdir(train_file_dir) if os.path.isdir(os.path.join(train_file_dir, c)) and not c.endswith('Set') and not c.startswith('.')]
class_list.sort()
print(class_list)
train_image_dir = os.path.join(all_file_dir, "trainImageSet")
if not os.path.exists(train_image_dir):
    os.makedirs(train_image_dir)
    
test_image_dir = os.path.join(all_file_dir, "testImageSet")
if not os.path.exists(test_image_dir):
    os.makedirs(test_image_dir)

val_image_dir = os.path.join(all_file_dir, "valImageSet")
if not os.path.exists(val_image_dir):
    os.makedirs(val_image_dir)
    
train_file = open("./Data/Data/train.csv", 'w')
test_file = open(os.path.join(all_file_dir, "test.csv"), 'w')
val_file = open(os.path.join(all_file_dir, "val.csv"), 'w')

with codecs.open(os.path.join(all_file_dir, "label_list.csv"), "w") as label_list:
    #加入训练集
    label_id = 0
    for class_dir in class_list:
        csv.writer(label_list).writerow([label_id, class_dir])
        image_path_pre = os.path.join(train_file_dir, class_dir)
                # 存在一些文件打不开，此处需要稍作清洗
        label_id += 1
    label_id = 0
    for class_dir in class_list:
        print("1")
        image_path_pre = os.path.join(train_file_dir, class_dir)
        for file in os.listdir(image_path_pre):
            try:
                img = Image.open(os.path.join(image_path_pre, file))
                shutil.copyfile(os.path.join(image_path_pre, file), os.path.join(train_image_dir, file))
                if "bacteria" in file:
                    label_id = 1
                if "virus" in file:
                    label_id = 2
                csv.writer(train_file).writerow([os.path.join(train_image_dir, file), label_id])
            except Exception as e:
                pass
    ##加入测试集
    label_id = 0
    for class_dir in class_list:
        image_path_pre = os.path.join(test_file_dir, class_dir)
        for file in os.listdir(image_path_pre):
            try:
                img = Image.open(os.path.join(image_path_pre, file))
                shutil.copyfile(os.path.join(image_path_pre, file), os.path.join(test_image_dir, file))
                if "bacteria" in file:
                    label_id = 1
                if "virus" in file:
                    label_id = 2
                csv.writer(test_file).writerow([os.path.join(test_image_dir, file), label_id])
            except Exception as e:
                pass
                # 存在一些文件打不开，此处需要稍作清洗
    label_id = 0
    for class_dir in class_list:
        image_path_pre = os.path.join(val_file_dir, class_dir)
        for file in os.listdir(image_path_pre):
            try:
                img = Image.open(os.path.join(image_path_pre, file))
                shutil.copyfile(os.path.join(image_path_pre, file), os.path.join(val_image_dir, file))
                if "bacteria" in file:
                    label_id = 1
                if "virus" in file:
                    label_id = 2
                csv.writer(val_file).writerow([os.path.join(val_image_dir, file), label_id])
            except Exception as e:
                pass
            
train_file.close()
test_file.close()  
val_file.close()  """

import torch
import pandas as pd
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
from torch import nn
from torch.nn import Conv2d
from torch.nn import Dropout
import copy 

class MyDataSet(Dataset):
    def __init__(self,image_path,csv_path,transforms,phrase):
        # 读取csv
        csv=pd.read_csv(csv_path,header=None)
        # 读取第一列,组合成完整的图片地址
        self.imgs=[str(k) for k in csv[0].values]
        self.phrase=phrase
        if self.phrase!="test":
            self.labels=np.asarray([k for k in csv[1].values])
        self.transforms=transforms
    

    def __getitem__(self, index):
        img_path = os.path.join("D:\学习\专业课\机器学习基础\肺炎分类" , self.imgs[1:index])
        pil_img = Image.open(img_path).convert("RGB")
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img)
            data = torch.from_numpy(pil_img)
        if self.phrase!="test":
            label=self.labels[index]
            sample = (data,label)
        else:
            sample=data
        return sample

    def __len__(self):
        return len(self.imgs)
from torchvision import transforms
# 数据增强
data_transforms={
    "train":
    transforms.Compose([
            transforms.Resize(size=(240,240)),
            transforms.CenterCrop(size=(224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "val":
    transforms.Compose([
            transforms.Resize(size=(240,240)),
            transforms.CenterCrop(size=(224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "test":
    transforms.Compose([
            transforms.Resize(size=(240,240)),
            transforms.CenterCrop(size=(224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    
}

class Embeddings(nn.Module):
    '''
    对图像进行编码，把图片当做一个句子，把图片分割成块，每一块表示一个单词
    '''
    def __init__(self,config,img_size,in_channels=3):
        super(Embeddings,self).__init__()
        img_size=img_size#224
        patch_size=config.patches["size"]#16
        ##将图片分割成多少块（224/16）*（224/16）=196
        n_patches=(img_size//patch_size)*(img_size//patch_size)
        #对图片进行卷积获取图片的块，并且将每一块映射成config.hidden_size维（768）
        self.patch_embeddings=Conv2d(in_channels=in_channels,
                                     out_channels=config.hidden_size,
                                     kernel_size=patch_size,
                                     stride=patch_size)
        
        #设置可学习的位置编码信息，（1,196+1,786）
        self.position_embeddings=nn.Parameter(torch.zeros(1,
                                                          n_patches+1,
                                                          config.hidden_size))
        #设置可学习的分类信息的维度
        self.classifer_token=nn.Parameter(torch.zeros(1,1,config.hidden_size))
        self.dropout=Dropout((config.transformer["dropout_rate"]))

    def forward(self,x):
        bs=x.shape[0]
        #cls_tokens=self.classifer_token.expand(bs,-1,-1)(bs,1,768)
        x=self.patch_embeddings(x)#（bs,768,14,14）
        x=x.flatten(2)#(bs,768,196)
        x=x.transpose(-1,-2)#(bs,196,768)
        cls_tokens=self.classifer_token.expand(bs,-1,-1)
        x=torch.cat((cls_tokens,x),dim=1)#将分类信息与图片块进行拼接（bs,197,768）
        embeddings=x+self.position_embeddings#将图片块信息和对其位置信息进行相加(bs,197,768)
        embeddings=self.dropout(embeddings)
        return  embeddings
# --------------------------------------- #
#（1）patch embedding
'''
img_size=224 : 输入图像的宽高
patch_size=16 ： 每个patch的宽高，也是卷积核的尺寸和步长
in_c=3 ： 输入图像的通道数
embed_dim=768 ： 卷积输出通道数
'''
# --------------------------------------- #
class patchembed(nn.Module):
    # 初始化
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768):
        super(patchembed, self).__init__()
        
        # 输入图像的尺寸224*224
        self.img_size = (img_size, img_size)
        # 每个patch的大小16*16
        self.patch_size = (patch_size, patch_size)
        # 将输入图像划分成14*14个patch
        self.grid_size = (img_size//patch_size, img_size//patch_size)
        # 一共有14*14个patch
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        # 使用16*16的卷积切分图像，将图像分成14*14个
        self.proj = nn.Conv2d(in_channels=in_c, out_channels=embed_dim, 
                              kernel_size=patch_size, stride=patch_size)
        
        # 定义标准化方法，给LN传入默认参数eps
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(embed_dim)
        
        
    # 前向传播
    def forward(self, inputs):
        # 获得输入图像的shape
        B, C, H, W = inputs.shape
        
        # 如果输入图像的宽高不等于224*224就报错
        assert H==self.img_size[0] and W==self.img_size[1], 'input shape does not match 224*224'
        
        # 卷积层切分patch [b,3,224,224]==>[b,768,14,14]
        x = self.proj(inputs)
        # 展平 [b,768,14,14]==>[b,768,14*14]
        x = x.flatten(start_dim=2, end_dim=-1)  # 将索引为 start_dim 和 end_dim 之间（包括该位置）的数量相乘
        # 维度调整 [b,768,14*14]==>[b,14*14,768]
        x = x.transpose(1, 2)  # 实现一个张量的两个轴之间的维度转换
        # 标准化
        x = self.norm(x)
        
        return x

#2.构建self-Attention模块
class Attention(nn.Module):
    def __init__(self,config,vis):
        super(Attention,self).__init__()
        self.vis=vis
        self.num_attention_heads=config.transformer["num_heads"]#12
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)  # 768/12=64
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 12*64=768

        self.query = Linear(config.hidden_size, self.all_head_size)#wm,768->768，Wq矩阵为（768,768）
        self.key = Linear(config.hidden_size, self.all_head_size)#wm,768->768,Wk矩阵为（768,768）
        self.value = Linear(config.hidden_size, self.all_head_size)#wm,768->768,Wv矩阵为（768,768）
        self.out = Linear(config.hidden_size, config.hidden_size)  # wm,768->768
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
        self.num_attention_heads, self.attention_head_size)  # wm,(bs,197)+(12,64)=(bs,197,12,64)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # wm,(bs,12,197,64)

    def forward(self, hidden_states):
        # hidden_states为：(bs,197,768)
        mixed_query_layer = self.query(hidden_states)#wm,768->768
        mixed_key_layer = self.key(hidden_states)#wm,768->768
        mixed_value_layer = self.value(hidden_states)#wm,768->768

        query_layer = self.transpose_for_scores(mixed_query_layer)#wm，(bs,12,197,64)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))#将q向量和k向量进行相乘（bs,12,197,197)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)#将结果除以向量维数的开方
        attention_probs = self.softmax(attention_scores)#将得到的分数进行softmax,得到概率
        weights = attention_probs if self.vis else None#wm,实际上就是权重
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)#将概率与内容向量相乘
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)#wm,(bs,197)+(768,)=(bs,197,768)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights#wm,(bs,197,768),(bs,197,197)

#3.构建前向传播神经网络
#两个全连接神经网络，中间加了激活函数
class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])#wm,786->3072
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)#wm,3072->786
        self.act_fn = torch.nn.functional.gelu#wm,激活函数
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)#wm,786->3072
        x = self.act_fn(x)#激活函数
        x = self.dropout(x)#wm,丢弃
        x = self.fc2(x)#wm3072->786
        x = self.dropout(x)
        return x

#4.构建编码器的可重复利用的Block()模块：每一个block包含了self-attention模块和MLP模块
class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size#wm,768
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)#wm，层归一化
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h#残差结构

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h#残差结构
        return x, weights

#5.构建Encoder模块，该模块实际上就是堆叠N个Block模块
class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

#6构建transformers完整结构，首先图片被embedding模块编码成序列数据，然后送入Encoder中进行编码
class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)#wm,对一幅图片进行切块编码，得到的是（bs,n_patch+1（196）,每一块的维度（768））
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)#wm,输出的是（bs,196,768)
        encoded, attn_weights = self.encoder(embedding_output)#wm,输入的是（bs,196,768)
        return encoded, attn_weights#输出的是（bs,197,768）

#7构建VisionTransformer，用于图像分类
class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, vis)
        self.head = Linear(config.hidden_size, num_classes)#wm,768-->10

    def forward(self, x, labels=None):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        #如果传入真实标签，就直接计算损失值
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return logits, attn_weights

import ml_collections
import argparse
import torch
import os
import numpy as np
from torch.utils.data import DataLoader

def get_config():
    '''
    配置transformer的模型的参数
    '''
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size':16})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


def save_model(args, model,epoch_index):
    '''
    保存每个epoch训练的模型
    '''
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "epoch%s_checkpoint.bin" % epoch_index)
    torch.save(model_to_save.state_dict(), model_checkpoint)



#实例化模型
def getVisionTransformers_model(args):
    config=get_config()#获取模型的配置文件
    num_classes = 3
    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    model.to(args.device)
    return args,model


#用测试集评估模型的训练好坏
def eval(args,model,test_loader):
    eval_loss=0.0
    total_acc=0.0
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    for i,batch in enumerate(test_loader):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits,_= model(x)#model返回的是（bs,num_classes）和weight
            batch_loss=loss_function(logits,y)
            #记录误差
            eval_loss+=batch_loss.item()
            #记录准确率
            _,preds= logits.max(1)
            num_correct=(preds==y).sum().item()
            total_acc+=num_correct

    loss=eval_loss/len(test_loader)
    acc=total_acc/(len(test_loader)*args.eval_batch_size)
    return loss,acc


def train(args,model):
    print("load dataset.........................")
    #加载数据

    val=MyDataSet("test/","D:\学习\专业课\机器学习基础\肺炎分类\Data\Data/test.csv",data_transforms["test"],"test")
    # 验证集比例
    indices = list(range(len(val)))
    random_seed= 42

    train_loader = DataLoader(val, batch_size=32, shuffle=True)
    val_loader=DataLoader(val, batch_size=16)
    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)

    print("training.........................")
    #设置测试损失list,和测试acc 列表
    val_loss_list=[]
    loaded_checkpoint = torch.load("D:\学习\专业课\机器学习基础\肺炎分类/output/epoch6_checkpoint.bin")
    val_acc_list=[]
    model.load_state_dict(loaded_checkpoint)
    #设置训练损失list
        # 每训练一个epoch,用当前训练的模型对验证集进行测试
    eval_loss, eval_acc = eval(args, model, val_loader)
    np.savetxt("val_loss_list.txt",val_loss_list)
    np.savetxt("val_acc_list.txt",val_acc_list)
    print("loss:{},acc:{}".format(eval_loss, eval_acc))

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    """parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10",
                        help="Which downstream task.")"""
    parser.add_argument("--output_dir", default="./output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--img_size", default=224, type=int,help="Resolution size")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--total_epoch", default=10, type=int,
                        help="Total number of training epochs to perform.")

    args = parser.parse_args()
    device = torch.device("cpu")
    args.device = device

    args,modle=getVisionTransformers_model(args)
    start=time.perf_counter()
    train(args,modle)
    end=time.perf_counter()
    print(end-start)

if __name__ == "__main__":
    main()

"""from torch.utils.data import DataLoader,SubsetRandomSampler
import numpy as np
batch_size=64
dataset=MyDataSet("train/","./Data/Data/train.csv",data_transforms["train"],"train")
indices = list(range(len(dataset)))
shuffle_dataset = True
random_seed= 42
batch_size=128
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)

val=MyDataSet("val/","./Data/Data/val.csv",data_transforms["val"],"val")
# 验证集比例
indices = list(range(len(val)))
random_seed= 42
batch_size=128
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_sampler = SubsetRandomSampler(dataset)
val_sampler = SubsetRandomSampler(val)

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_dataset)
val_loader=DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_dataset)"""


