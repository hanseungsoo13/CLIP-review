# Image Encoder: ResNet 50
# replace global average pooling layer with an attention pooling mechanism

# Text Encoder: Transformer
# 63M parameter 12-layer, 512 wide model with 8-attention head
# BPE representation 49152 vocab size
# max sequence: 76
# start: [SOS], end: [EOS] 
# highest layer of the transformer at [EOS]: feature representation
# -> layer normalized and linearly projected into the multi-modal embedding space
# Masked self-attention was used in the text encoder

# 32 epoch
# Adam optimzer
# decoupled weight decay regularization
# cosine schedule


# objectives: ulti-calss N-pair loss

# augmentation: random square crop

import torch
from torch import nn
import Resnet50
import transformer
import numpy as np


class CLIP(nn.Module):
    def __init__(self,
                 #image
                 v_layers,v_heads:int,v_input_size,v_emb_size,
                 #text
                 max_length,vocab_size,n_layers,model_dim, attn_heads):
        super().__init__()

        self.max_length=max_length

        self.visual = Resnet50.ModifiedResNet(
            layers = v_layers,
            heads=v_heads,
            input_size=v_input_size,
            emb_size = v_emb_size
        )

        self.transformer = transformer.Transformer(
            max_length=max_length,
            vocab_size=vocab_size,
            n_layers=n_layers,
            model_dim=model_dim,
            attn_heads=attn_heads,
            attn_mask=self.build_attention_mask()
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))



    def build_attention_mask(self):
        mask = torch.empty(self.max_length,self.max_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask
    

    def forward(self,image,text):
        #text_token = self.tokenizer(text)
        image_features = self.visual(image)
        text_features = self.transformer(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.permute([0,2,1])
        logits_per_text = logits_per_image.permute([0,2,1])

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text