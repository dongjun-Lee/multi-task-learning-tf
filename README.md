# Multi-task Learning with TensorFlow
Tensorflow implementation of multi-task learning. (Language model & Text classification)

## Model
<img src="https://user-images.githubusercontent.com/6512394/44083292-884d66b4-9fee-11e8-9e36-d59ef27ae51c.PNG">

## Requirements
- Python3
- TensorFlow
- pip install -r requirements.txt

## Usage

```
$ python train.py
```

### Hyperparameters
```
$ python train.py -h
usage: train.py [-h] [--embedding_size EMBEDDING_SIZE]
                [--num_layers NUM_LAYERS] [--num_hidden NUM_HIDDEN]
                [--keep_prob KEEP_PROB] [--learning_rate LEARNING_RATE]
                [--batch_size BATCH_SIZE] [--num_epochs NUM_EPOCHS]
                [--max_document_len MAX_DOCUMENT_LEN]

optional arguments:
  -h, --help            show this help message and exit
  --embedding_size EMBEDDING_SIZE
                        embedding size.
  --num_layers NUM_LAYERS
                        RNN network depth.
  --num_hidden NUM_HIDDEN
                        RNN network size.
  --keep_prob KEEP_PROB
                        dropout keep prob.
  --learning_rate LEARNING_RATE
                        learning rate.
  --batch_size BATCH_SIZE
                        batch size.
  --num_epochs NUM_EPOCHS
                        number of epochs.
  --max_document_len MAX_DOCUMENT_LEN
                        max document length.
```

## Experimental Results

### Language Model Training Loss
<img src="https://user-images.githubusercontent.com/6512394/44083279-8134afe0-9fee-11e8-8ff8-7cd93b32001c.PNG">

### Text Classification Training Loss
<img src="https://user-images.githubusercontent.com/6512394/44083286-84b0faac-9fee-11e8-81cf-cd1327cbb43e.PNG">