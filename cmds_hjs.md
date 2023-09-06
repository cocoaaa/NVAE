# Prepare for lmdb dataset for celeba64
cd $CODE_DIR/scripts
export DATA_ROOT='/data/datasets/reverse-eng-data/originals'

## train:
1. set the conda environment to torch160 (? #to-verify)
conda activate torch160
2. run the training command
python create_celeba64_lmdb.py --split train \
--img_path $DATA_ROOT --lmdb_path $DATA_ROOT/Celeba_lmdb
- status: ran successfully

## valid
python create_celeba64_lmdb.py --split valid \
--img_path $DATA_ROOT --lmdb_path $DATA_ROOT/Celeba_lmdb


## test
python create_celeba64_lmdb.py --split test \
--img_path $DATA_ROOT --lmdb_path $DATA_ROOT/Celeba_lmdb
- status: ran successfully

## Summary on creating lmdb for celeba64
Now, each [train/valid/test] lmdb database is created under the root dir:
`/data/datasets/reverse-eng-data/originals/Celeba_lmdb`:
- train: `/data/datasets/reverse-eng-data/originals/Celeba_lmdb/train.lmdb`
- valid: `/data/datasets/reverse-eng-data/originals/Celeba_lmdb/valid.lmdb`
- test:  `/data/datasets/reverse-eng-data/originals/Celeba_lmdb/test.lmdb`
---
# Train a model with a new data-preprocessing
- The checkpoint provided by the authors do not apply any center-crop before resizing the original celeba images to 64x64.
- For consistency in our generative models in our GM dataset, we add the center-crop at 140 (and then resize to 64) to the data prep pipeline. 
- With this new data-preprocessing, we train a new nvae model

## Cmd to run training
### Note
- for data input:
  - DATA_DIR:  location of the data corpus
    - for celeba_64 lmdb data-read: it should be the directory containing `train.lmdb' and 'validation.lmdb')
    ```python
      train_data = LMDBDataset(root=args.data, name='celeba64', train=True, transform=train_transform, is_encoded=True)
      valid_data = LMDBDataset(root=args.data, name='celeba64', train=False, transform=valid_transform, is_encoded=True)
    ```

- for gpu processor setup:
  - num_process_per_node: number of gpus; defualt is 1
  - num_nf: number of normalizing flow cells per groups; see to zero to disablbe flows; default is 0

- for logging and saving results (e.g., checkpoints, logs)
  - root: location of the results 
  - save: id used for stroing interprediate results
  - note that:
    -  args.save is right away set to args.root / 'eval-' + args.save
    - and then, logger is set to args.save directory
      - logger writes to args.save/ 'log.txt' file
      - writer writes Tensorboard files with log_dir=args.save


### training cmd for a single gpu
export CUDA_VISIBLE_DEVICES=0
export NGPUS=1
export EXPR_ID=2022-0528-1710
export DATA_DIR=/data/datasets/reverse-eng-data/originals/Celeba_lmdb
export CHECKPOINT_DIR='./Training-Outputs'

nohup python train.py --data $DATA_DIR --root $CHECKPOINT_DIR --save $EXPR_ID --dataset celeba_64 \
        --num_channels_enc 64 --num_channels_dec 64 --epochs 90 --num_postprocess_cells 2 --num_preprocess_cells 2 \
        --num_latent_scales 3 --num_latent_per_group 20 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 \
        --num_preprocess_blocks 1 --num_postprocess_blocks 1 --weight_decay_norm 1e-1 --num_groups_per_scale 20 \
        --batch_size 16 --num_nf 1 --ada_groups --num_process_per_node $NGPUS --use_se --res_dist --fast_adamax &

- run_id: 
- started:
- gpus: 
- pids: 


### training cmd for multiple gpus
export CUDA_VISIBLE_DEVICES=0,1,2
export NGPUS=3
export BS=4
export EXPR_ID=2022-0528-1720
export DATA_DIR=/data/datasets/reverse-eng-data/originals/Celeba_lmdb
export CHECKPOINT_DIR='./Training-Outputs'

nohup python train.py --data $DATA_DIR --root $CHECKPOINT_DIR --save $EXPR_ID --dataset celeba_64 \
        --num_channels_enc 64 --num_channels_dec 64 --epochs 90 --num_postprocess_cells 2 --num_preprocess_cells 2 \
        --num_latent_scales 3 --num_latent_per_group 20 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 \
        --num_preprocess_blocks 1 --num_postprocess_blocks 1 --weight_decay_norm 1e-1 --num_groups_per_scale 20 \
        --batch_size $BS --num_nf 1 --ada_groups --num_process_per_node $NGPUS --use_se --res_dist --fast_adamax &

- run_id: EXPR_ID=2022-0528-1720
- started: 5:20pm 05-28-2022
- gpus: 0,1,2
- pids: 24812, 24813, 24814

---
# Sampling from a checkpoint released by authors
Load pretrained model and sample images from the model

## Set variables to use in command
- STEMP is the sampling temperature

export CUDA_VISIBLE_DEVICES=3
export CKPT_FP='./Outputs/Checkpoints/nvae_celeba64_checkpoint.pt'
export STEMP=0.8

## Run sampling command
 nohup python evaluate.py --checkpoint $CKPT_FP --eval_mode sample --temp $STEMP --save ./Outputs/Samples-NPZ-$STEMP  --readjust_bn --num_samples 128000 &
 
 
---
# Sampling from ckpts of my own training runs
- date: 2022-09-09 18:05

## sample using `Training-Outputs/eval-2022-0528-1720/checkpoint.pt`
export STEMP=0.8
export CKPT_FP='./Training-Outputs/eval-2022-0528-1720/checkpoint.pt'
export SAVE_DIR="./Training-Outputs/eval-2022-0528-1720/image_samples_$STEMP"

CUDA_VISIBLE_DEVICES=0 nohup python evaluate.py --checkpoint $CKPT_FP --eval_mode sample --temp $STEMP --save $SAVE_DIR  --readjust_bn --num_samples 128000 &
 
 - Status:
   - started: 9/9/22, 6:13 PM
   - pid: 13220
   - gpu: 0
   - samples are being saved to: "./Training-Outputs/eval-2022-0528-1720/image_samples_0.8" 



# Command log for generating images from model trained on CelebA-HQ 256
- date: 20230228-180415
- purpose: to create GM256 dataset for gm fingerprinting (iccv23)

## Steps:
- [ ] download their released ckpt for celeba hq 256: [link](https://drive.google.com/drive/folders/1KVpw12AzdVjvbfEYM_6_3sxTy93wWkbe)
- [ ] move the ckpt file(s) to arya: `./ckpts/celeba_256/qualitative/checkpoint.pt` or `./ckpts/celeba_256/quantitave/checkpoint.pt`
- [ ] run the sampling by following my notes (used for sampling from celeba64 trained models):

## 1. Using released ckpt (qualitative) - authors say this ckpt generates images with better qualitative realism even though achieves lower fid score.
todo: 
- [ ] try: sample temp of 0.6, 0.7, 0.8

### Sample w/ temp=0.8
- reduced the sampling batch size to 16 (from 100) for memory limit

conda activate torch160
export STEMP=0.8 
export CKPT_FP="./ckpts/celeba_256/qualitative/checkpoint.pt"
export MODEL_NAME="celeba_256_qual"
export SAVE_DIR="./Samples/$MODEL_NAME_$STEMP"

CUDA_VISIBLE_DEVICES=1 nohup python evaluate.py --checkpoint $CKPT_FP --eval_mode sample \
   --temp $STEMP --save $SAVE_DIR  --readjust_bn --num_samples 100000 &
 
 - Status:
   - started: 20230228-18451
   - pid: 2943 
   - gpu: 1
   - samples are being saved to: `samples/celeba_256/qualitative` or  `samples/celeba_256/qualitative`


1. Using released ckpt (quantitative) 
