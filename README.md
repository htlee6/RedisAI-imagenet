# RedisAI-imagenet

This repo demonstrates a simple example of how to use RedisAI to classify ImageNet images using a pre-trained model with RedisAI.

## Dependencies
- Redis
- RedisAI (`Docker` container)
- RedisAI-py
```bash
pip install redisai==1.3.0
```
- pytorch
```bash
pip install torch==1.11.0
```
- torchvision
```bash
pip install torchvision==0.12.0
```

### Tutorial Post

https://htlee.github.io/blog/2023/07/20/RedisAI-Example/

### Run the app
```bash
python app.py --image img/dog.jpeg --model_name resnet50
```