# VisualReasoning_MMnet
Models in Pytorch for visual reasoning task on Clevr dataset. <br><br>
**Stack attention**:<br>
https://arxiv.org/pdf/1511.02274.pdf<br><br>
**Module network**:<br>
https://arxiv.org/pdf/1705.03633.pdf<br><br>
Yes, but what's new? <br>
Try to archive same performance in end-to-end differentiable architecture: <br>
**Module memory network** [new]<br>
**Module memory network end2end differentiable** [new] <br><br>
Try to archive weak supervision: <br>
(**Work in progress**) <br>

## Set-up
### Step 1: Download the data
```
mkdir data
wget https://s3-us-west-1.amazonaws.com/clevr/CLEVR_v1.0.zip -O data/CLEVR_v1.0.zip
unzip data/CLEVR_v1.0.zip -d data
```
### Step 2: Extract Image Features
```
python scripts/extract_features.py \
  --input_image_dir data/CLEVR_v1.0/images/train \
  --output_h5_file data/train_features.h5
```
### Step 3: Preprocess Questions
```
python scripts/preprocess_questions.py \
  --input_questions_json data/CLEVR_v1.0/questions/CLEVR_train_questions.json \
  --output_h5_file data/train_questions.h5 \
  --output_vocab_json data/vocab.json
```

### Test sample
<img src="pics/pg_sample.jpg" width="300">

## Train
```
python train.py [-args]

arguments:
  --model               Model to train: SAN, SAN_wbw, PG, PG_memory, PG_endtoend
  --question_size       Number of words in question dictionary
  --stem-dim            Number of feature-maps
  --n-channel           Number of features channels
  --batch_size          Mini-batch dim
  --min_grad            Minimum value of gradient clipping
  --max_grad            Maximum value of gradient clipping
  --load_model_path     Load pre-trained model (path)
  --load_model_mode     Load model mode: Execution engine (EE), Program Generator (PG), Both (PG+EE)
  --save_model          Save model ? (bool)
  --clevr_dataset       Clevr dataset data (path)
  --clevr_val_images    Clevr dataset validation images (path)
  --num_iterations      Num iteration per epoch
  --num_val_samples     Number validation samples
  --batch_multiplier    Virtual batch (minimum value: 1)
  --train_mode          Train mode:  Execution engine (EE), Program Generator (PG), Both (PG+EE)
  --decoder_mode        Progam generator mode: Backpropagation (soft, gumbel) Reinforce (hard, hard+penalty)
  --use_curriculum      Use curriculum to train program generator (bool) 
```


Module memory network (Pg_memory)<br>
<img src="pics/loss_pg_memory.jpg" width="600"><br><br>
Module memory network end2end (Pg_endtoend)<br>
<img src="pics/loss_pg_endtoend.jpg" width="600"><br>

## Models<br>

### Stack Attention (SAN)
<img src="pics/san.jpg" width="700">

### Stack Attention word2word (SAN_wbw)
<img src="pics/san_w2w.jpg" width="700">

### Module Network (PG)
<img src="pics/pg.jpg" width="500">

### Module-Memory Network (PG_memory)
<img src="pics/mmNet.jpg" width="700">

### Module-Memory Network end2end (PG_endtoend)
<img src="pics/mmNet_endtoend.jpg" width="850">




