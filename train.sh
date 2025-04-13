python train.py --model_name resnext50_32x4d --model_type resnext50_32x4d --batch_size 32 --num_epochs 10
python train.py --model_name resnext101_32x8d --model_type resnext101_32x8d --batch_size 32 --num_epochs 30

python inference.py --model_type resnext50_32x4d --model_path ./checkpoints/resnext50_32x4d.pth --output_dir resnext50_32x4d --batch_size 32