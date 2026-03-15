export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

compute-sanitizer --tool memcheck --track-unused-memory torchrun \
    --master_port 11397\
    --nproc_per_node=4 train.py \
    --sync-bn \
    >train_radarrwkv.log 2>&1
