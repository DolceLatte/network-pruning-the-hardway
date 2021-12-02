# network-pruning-the-hardway
Network pruning을 직접 구현하고 torch_pruning 패키지와 성능을 비교합니다. 

```python
pip install torch_pruning
```

## Pruning VGG 11
![image](https://user-images.githubusercontent.com/45285053/144351606-c6a73c15-6fee-47c8-88ca-d1020ff0e43e.png)

### evaluation metric
-  추론 시간 대비 정확도 및 학습 파라미터 수 대비 정확도

#### reference 
- https://github.com/yasersakkaf/Neural-Network-Pruning-using-PyTorch
- https://github.com/VainF/Torch-Pruning
