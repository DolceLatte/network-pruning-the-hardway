# network-pruning-the-hardway
Network pruning을 직접 구현하고 torch_pruning 패키지와 성능을 비교합니다. 


#### train vgg 11
```bash
python3 train.py # train vgg11
```

#### network-pruning-the-hardway
```bash
python3 pruning.py # after train vgg11
```

#### network-pruning-using torch_pruning
```bash
python3 ./script/quick_start_tp.py # after train vgg11
```

### evaluation metric
-  추론 시간 대비 정확도 및 학습 파라미터 수 대비 정확도

### reference 
- https://github.com/yasersakkaf/Neural-Network-Pruning-using-PyTorch
- https://github.com/VainF/Torch-Pruning
- ```python
pip install torch_pruning
```
