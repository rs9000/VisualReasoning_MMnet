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


## Test sample
<img src="pics/pg_sample.jpg" width="300">

## Train
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




