python MLP/regression.py      --batch_size=2048 --num_epochs=100 --train_percent=-1   --lrs="[1e-5, 1e-4, 1e-3, 1e-2, 1e-1]"
python MLP/classification.py  --batch_size=2048 --num_epochs=100 --train_percent=-1   --lrs="[1e-5, 1e-4, 1e-3, 1e-2, 1e-1]"
python MLP/regression.py      --batch_size=2048 --num_epochs=100 --train_percent=0.3  --lrs="[1e-5, 1e-4, 1e-3, 1e-2, 1e-1]"
python MLP/classification.py  --batch_size=2048 --num_epochs=100 --train_percent=0.3  --lrs="[1e-5, 1e-4, 1e-3, 1e-2, 1e-1]"