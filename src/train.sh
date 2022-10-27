git pull
source ~/miniconda/etc/profile.d/conda.sh
conda activate multi-objects
python3 train.py --batch_size 32 --seed 2 --path_to_dataset "./dataset/data/" --max_epochs 200 --devices 1 --lr 0.00005 --img_size 64
