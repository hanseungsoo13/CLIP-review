from CLIP import CLIP
import torch
from torchinfo import summary
from Resnet50 import ModifiedResNet, BottleNeck
from transformer import Transformer
from torchtext.transforms import CLIPTokenizer

device = "cuda" if torch.cuda.is_available() else 'cpu'

model = CLIP(v_layers=[3,6,3,2],v_heads=8,v_input_size=224,v_emb_size=64,
      max_length=10,vocab_size=5000,n_layers=12,model_dim=2048, attn_heads=8).to(device)

MERGES_FILE = "http://download.pytorch.org/models/text/clip_merges.bpe"
ENCODER_FILE = "http://download.pytorch.org/models/text/clip_encoder.json"
tokenizer = CLIPTokenizer(MERGES_FILE,ENCODER_FILE)
#t=torch.Tensor(tokenizer("An image of a dog")).to(device)
#print(t)
#visual_model = ModifiedResNet(layers=[3,6,3,2],heads=8,input_size=224,emb_size=64).to(device) #(1,2048)
#text_model = Transformer(10,5000,8,64,8,model.build_attention_mask()).to(device)

summary(model,[(1,3,224,224),(1,10)],dtypes=[torch.cuda.FloatTensor,torch.long],device=device)
#summary(visual_model,(1,3,224,224),dtypes=[torch.cuda.FloatTensor],device=device)
#summary(text_model,(1,10),dtypes=[torch.long],device=device)

#t = torch.rand([1,3,224,224]).to(device)

#out = model(t)
#print(out.shape)