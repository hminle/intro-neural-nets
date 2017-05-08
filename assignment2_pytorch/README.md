# CIFAR10 - Pytorch

## How to use main.py
usage: main.py [-h] [--lr LR] [--resume] [--epochs EPOCHS]
               [--batch_size BATCH_SIZE] [--pretrained]
               [--lr_decay_epoch LR_DECAY_EPOCH]

PyTorch CIFAR10 Training

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               learning rate
  --resume, -r          resume from checkpoint
  --epochs EPOCHS, -e EPOCHS
                        num of epochs
  --batch_size BATCH_SIZE, -b BATCH_SIZE
                        batch size
  --pretrained, -p      using pretrained model
  --lr_decay_epoch LR_DECAY_EPOCH
                        Specify the epoch when lr will be decayed

**Example**:
```
python main.py --epochs 250 --batch_size 16 --lr 0.1
```

```
python main.py --resume --batch_size 32 --lr 0.01
```

## How to use eval.py
usage: eval.py [-h] [--batch_size BATCH_SIZE] --checkpoint CHECKPOINT
               [--draw_confusion]

PyTorch CIFAR10 Training

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE, -b BATCH_SIZE
                        batch size
  --checkpoint CHECKPOINT, -c CHECKPOINT
                        checkpoint file path
  --draw_confusion, -d  draw confusion matrix

**Example**:
```
python eval.py --checkpoint ckpt_resnet18.t7_bk --batch_size 16 --draw_confusion 
```
