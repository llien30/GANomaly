# GANomaly
[arXiv](https://arxiv.org/abs/1805.06725)
[GitHub](https://github.com/samet-akcay/ganomaly)
[My Paper Summary](https://github.com/mn1204/paper_summary/issues/1)

## About CONFIG file

config file must be written in the following format

```.yaml
dataset: MNIST
abnormal_class: 0

save_dir: './weights'

batch_size: 64
test_batch_size: 64

channel: 1
input_size: 32
z_dim: 100
ndf: 64
ngf: 64
extralayers: 0

w_adv: 1
w_con: 50
w_enc: 1

num_epochs: 15

# the name this experiment on wandb
name: mnist
```
