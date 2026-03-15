# RadarRWKV: Dirichlet-guided Variational Sparse Bottleneck for 4D Radar Perception

The code implements the proposed algorithm and includes scripts for reproducing the experimental results.

### Environments
- Python 3.9
- PyTorch 2.4.1+cu118
- Use `pip install -r requirements.txt` to install dependencies.

### Datasets
- 4D Radar [RaDelft](https://github.com/RaDelft/RaDelft-Dataset)
- 4D Radar [RADIaL](https://github.com/valeoai/RADIal)

### Train
1. To train the model from scratch, run:
   ```python
   sh train.sh
   ```
2. To evaluate a pretrained model, download the checkpoint from and run:
   ```python
   sh evaluate.sh
   ```
