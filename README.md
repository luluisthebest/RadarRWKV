# RadarRWKV: Dirichlet-guided Variational Sparse Bottleneck for 4D Radar Perception

The code implements the proposed algorithm and includes scripts for reproducing the experimental results.

### Environments
- Python 3.9
- PyTorch 2.4.1+cu124
- Use `pip install -r requirements.txt` to install dependencies.

### Datasets
- 4D Radar [RaDelft](https://github.com/RaDelft/RaDelft-Dataset)
- 4D Radar [RADIaL](https://github.com/valeoai/RADIal)

### Train
1. Set "dataset_path" and "output" in `configs/RaDelft.json` to point to your local paths.
2. To train the model from scratch, run:
   ```python
   sh train.sh
   ```
3. To evaluate a pretrained model, download the checkpoint from [Google Drive](https://drive.google.com/file/d/113WHGf6JEjBV23Y1VOH2j41Oq31rBoX5/view?usp=drive_link) (SHA256 Checksum: FBC0020310FD95AC0D92F4B69C51696B5AFBE61A8D6B1F977E1DAEB1C8403F35) and run:
   ```python
   sh evaluate.sh
   ```
