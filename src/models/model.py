import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision


class MultiHeadAttention(nn.Module):
    r"""Multi-headed Attention for input Query, Key, Value
    Multi-headed Attention is a module for attention mechanisms which runs through attention in several times in
    parallel, then the multiple outputs are concatenated and linearly transformed

    Args:
        embed_size  (int): Max embedding size
        num_heads   (int): Number of heads in multi-headed attention; Number of splits in the embedding size
        dropout     (float, optional): Percentage of Dropout to be applied in range 0 <= dropout <=1
        batch_dim   (int, optional): The dimension in which batch dimensions is
    """

    def __init__(self, embed_size: int, num_heads: int, dropout: float = 0.2, batch_dim: int = 0):
        super(MultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_dim = batch_dim

        self.dropout_layer = nn.Dropout(dropout)

        self.head_size = self.embed_size // self.num_heads

        assert self.head_size * self.num_heads == self.embed_size, "Heads cannot split Embedding size equally"

        self.Q = nn.Linear(self.embed_size, self.embed_size)
        self.K = nn.Linear(self.embed_size, self.embed_size)
        self.V = nn.Linear(self.embed_size, self.embed_size)

        self.linear = nn.Linear(self.embed_size, self.embed_size)

    def forward(self, q, k, v, mask=None):
        q_batch_size, q_seq_len, q_embed_size = q.size()
        k_batch_size, k_seq_len, k_embed_size = k.size()
        v_batch_size, v_seq_len, v_embed_size = v.size()

        q = self.Q(q).reshape(q_batch_size, q_seq_len, self.num_heads, self.head_size)
        k = self.K(k).reshape(k_batch_size, k_seq_len, self.num_heads, self.head_size)
        v = self.V(v).reshape(v_batch_size, v_seq_len, self.num_heads, self.head_size)

        attention = self.attention(q, k, v, mask=mask)
        concatenated = attention.reshape(v_batch_size, -1, self.embed_size)
        out = self.linear(concatenated)

        return out

    def attention(self, q, k, v, mask=None):
        scores = torch.einsum("bqhe,bkhe->bhqk", [q, k])

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        scores /= math.sqrt(self.embed_size)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout_layer(scores)
        attention = torch.einsum("bhql,blhd->bqhd", [scores, v])
        return attention


class VisionEncoder(nn.Module):
    r"""Vision Encoder Model
        An Encoder Layer with the added functionality to encode important local structures of a tokenized image
        Args:
            embed_size      (int): Embedding Size of Input
            num_heads       (int): Number of heads in multi-headed attention
            hidden_size     (int): Number of hidden layers
            dropout         (float, optional): A probability from 0 to 1 which determines the dropout rate
    """

    def __init__(self, embed_size: int, num_heads: int, hidden_size: int, dropout: float = 0.1):
        super(VisionEncoder, self).__init__()

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.norm1 = nn.LayerNorm(self.embed_size)
        self.norm2 = nn.LayerNorm(self.embed_size)

        self.attention = MultiHeadAttention(self.embed_size, self.num_heads, dropout=dropout)

        self.mlp = nn.Sequential(
            nn.Linear(self.embed_size, 4 * self.embed_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(4 * self.embed_size, self.embed_size),
            nn.Dropout(self.dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        x = x + self.attention(x, x, x)
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    r"""Vision Transformer Model
        A transformer model to solve vision tasks by treating images as sequences of tokens.
        Args:
            image_size      (int): Size of input image
            channel_size    (int): Size of the channel
            patch_size      (int): Max patch size, determines number of split images/patches and token size
            embed_size      (int): Embedding size of input
            num_heads       (int): Number of heads in Multi-Headed Attention
            classes         (int): Number of classes for classification of data
            hidden_size     (int): Number of hidden layers
            dropout         (float, optional): A probability from 0 to 1 which determines the dropout rate
    """

    def __init__(self, image_size: int, channel_size: int, patch_size: int, embed_size: int, num_heads: int,
                 classes: int, num_layers: int, hidden_size: int, dropout: float = 0.1):
        super(ViT, self).__init__()

        self.p = patch_size
        self.image_size = image_size
        self.embed_size = embed_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = channel_size * (patch_size ** 2)
        self.num_heads = num_heads
        self.classes = classes
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)

        self.embeddings = nn.Linear(self.patch_size, self.embed_size)
        self.class_token = nn.Parameter(torch.randn(1, 1, self.embed_size))
        self.positional_encoding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.embed_size))

        self.encoders = nn.ModuleList([])
        for layer in range(self.num_layers):
            self.encoders.append(VisionEncoder(self.embed_size, self.num_heads, self.hidden_size, self.dropout))

        self.norm = nn.LayerNorm(self.embed_size)

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_size, self.classes)
        )

    def forward(self, x, mask=None):
        b, c, h, w = x.size()

        x = x.reshape(b, int((h / self.p) * (w / self.p)), c * self.p * self.p)
        x = self.embeddings(x)

        b, n, e = x.size()

        class_token = self.class_token.expand(b, 1, e)
        x = torch.cat((x, class_token), dim=1)
        x = self.dropout_layer(x + self.positional_encoding)

        for encoder in self.encoders:
            x = encoder(x)

        x = x[:, -1, :]

        x = self.classifier(self.norm(x))

        return x


class DeiT(nn.Module):
    r"""Data-efficient image Transformer (DeiT) Implementation
        The Data-efficient image Transformer (DeiT) is for multi-class image classification which is trained through
        data distillation
        Args:
            image_size      (int): Input Image height/width size
            channel_size    (int): Number of Channels in Input Image
            patch_size      (int): Size of Each Patch for Input Image
            embed_size      (int): Max embedding size
            num_heads       (int): Number of heads in multi-headed attention
            classes         (int): Number in of distinct classes for classification
            num_layers      (int): Number of encoder blocks in DeiT
            hidden_size     (int): Number of hidden units in feed forward of encoder
            teacher_model   (object): Teacher model for data distillation
            dropout         (float, optional): A probability from 0 to 1 which determines the dropout rate
    """

    def __init__(self, image_size: int, channel_size: int, patch_size: int, embed_size: int, num_heads: int,
                 classes: int, num_layers: int, hidden_size: int, teacher_model, dropout: float = 0.1):
        super(DeiT, self).__init__()

        self.image_size = image_size
        self.channel_size = channel_size
        self.p = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = channel_size * (patch_size ** 2)
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.classes = classes
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.dropout_layer = nn.Dropout(self.dropout)

        self.norm = nn.LayerNorm(self.embed_size)

        self.embeddings = nn.Linear(self.patch_size, self.embed_size)
        self.class_token = nn.Parameter(torch.randn(1, 1, self.embed_size))
        self.distillation_token = nn.Parameter(torch.randn(1, 1, self.embed_size))
        self.positional_encoding = nn.Parameter(torch.randn(1, self.num_patches + 2, self.embed_size))

        self.teacher_model = None #teacher_model
        #for parameter in self.teacher_model.parameters():
        #    parameter.requires_grad = False
        #self.teacher_model.eval()

        self.encoders = nn.ModuleList([])
        for layer in range(self.num_layers):
            self.encoders.append(VisionEncoder(self.embed_size, self.num_heads, self.hidden_size, self.dropout))

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_size, self.classes)
        )

    def forward(self, x, mask=None):
        b, c, h, w = x.size()

        #teacher_logits_vector = self.teacher_model(x)

        x = x.reshape(b, int((h / self.p) * (w / self.p)), c * self.p * self.p)
        x = self.embeddings(x)

        b, n, e = x.size()

        class_token = self.class_token.expand(b, 1, e)
        x = torch.cat((class_token, x), dim=1)

        distillation_token = self.class_token.expand(b, 1, e)
        x = torch.cat((x, distillation_token), dim=1)

        x = self.dropout_layer(x + self.positional_encoding)

        for encoder in self.encoders:
            x = encoder(x)

        x, distillation_token = x[:, 0, :], x[:, -1, :]

        x = self.classifier(self.norm(x))

        return x  # , distillation_token


class VGG16_classifier(nn.Module):
    def __init__(self, classes, hidden_size, img_size_preprocess=224, preprocess_flag=False, dropout=0.1):
        super(VGG16_classifier, self).__init__()

        self.classes = classes
        self.hidden_size = hidden_size
        self.img_size_preprocess = img_size_preprocess
        self.preprocess_flag = preprocess_flag
        self.dropout = dropout

        self.vgg16 = torchvision.models.vgg16(pretrained=True)

        for parameter in self.vgg16.parameters():
            parameter.requires_grad = True

        self.preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(self.img_size_preprocess, self.img_size_preprocess)),
            torchvision.transforms.ToTensor()
        ])

        self.vgg16.classifier = nn.Sequential(
            nn.Linear(25088, self.hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size * 4, self.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.classes)
        )

    def forward(self, x):
        if self.preprocess_flag:
            x = self.preprocess(x)
        x = self.vgg16(x)
        return x
