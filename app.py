import argparse
import numpy as np
import pandas as pd
from inference import pred
import torch 
import json 

# load labels from json file
with open('labels.json', 'r') as f:
    labels = np.array(json.load(f))

# parse input image path
parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default='img/panda.jpg', help='path to image')
parser.add_argument('--model_name', type=str, default='mobilenet_v3_small', help='model name')
args=parser.parse_args()

# run inference
op = pred(args.image, model_name=args.model_name)
topk = torch.topk(torch.Tensor(op), k=5, dim=1) # get top 5 results
scores, indices = topk.values.squeeze().numpy(), topk.indices.to(torch.int64).squeeze().numpy()

# show result scores and labels
df = pd.DataFrame(np.transpose([labels[indices], scores]), columns=['label', 'score'])
print(df.to_string(index=False))