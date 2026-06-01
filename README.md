From scratch implementation of several diffusion architectures; [DDPM](https://arxiv.org/pdf/2006.11239), [ADM](https://arxiv.org/pdf/2105.05233), and [DiT](https://arxiv.org/pdf/2212.09748)


## Examples
Here are examples of how to trigger a training run:

### ADM DDPM style CIFAR10 32x32 images:
```bash
python3 main.py --device cuda --dataset cifar10 --total_steps 800000 --batch_size 128 --grad_accum 1
```

### DiT on CelebA 32x32 images with 4 patches:
```bash
python3 main.py --device cuda --dataset celeb_small --total_steps 800000 --batch_size 64 --grad_accum 2 --model_name dit --patch_size 4
```

## To-Do:
- Add DDIM for faster inference
- Wandb integration with sampled images and FID score
- Improved DDPM for hybrid loss and capability to learn variance (DiT variance head is currently left untrained)
- Classifier-free diffusion guidance training to make use of class labels
- Training warmup
- Migrating to YAML config instead of cli args
- For eval metrics for validation set; FID, sFID, IS, precision/recall
- LSUN dataset variant

There is probably more, but these are top of mind