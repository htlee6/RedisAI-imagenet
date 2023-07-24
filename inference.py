import numpy as np
from redisai import Client
import torch 
import torchvision as tv
import os
import logging
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop

def pred(imagepath: str, model_name: str = 'mobilenet_v3_small', host: str = 'localhost', port: int = 6379):
    model_filename = model_name+'.pt'

    con = Client(host=host, port=port)
    logging.log(logging.INFO, "RedisAI client connected to redis server")

    if not hasattr(tv.models, model_name):
        raise ValueError(f"Model {model_name} not found in torchvision.models")
    pretrain_model = getattr(tv.models, model_name)(pretrained=True)
    pretrain_model.eval()
    if not os.path.exists(model_filename):
        script_module = torch.jit.trace(pretrain_model, example_inputs=torch.randn(1,3,224,224))  # The model should be TorchScriptModel so that it can be loaded by RedisAI
        torch.jit.save(script_module, model_filename)

    model = open(model_filename, 'rb').read()
    logging.log(logging.INFO, "Model pre-loaded")
    
    con.modelset(model_name, 'torch', 'cpu', model, inputs=['input'], outputs=['pred_output'])
    logging.log(logging.INFO, "Model loaded into RedisAI")

    # load image from file to pytorch tensor for pretrained model from parser
    img = Image.open(imagepath)
    preprocess_transform = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(std=[0.229, 0.224, 0.225], mean=[0.485, 0.456, 0.406])
    ])
    img_tensor = preprocess_transform(img).unsqueeze(0)

    con.tensorset('input', img_tensor.numpy(), dtype='float')
    con.modelrun(model_name, inputs=['input'], outputs=['pred_output'])
    op = con.tensorget('pred_output')
    return op