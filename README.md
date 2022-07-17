# 第七届信也科技杯 RobertAckley 初赛代码
这是第七届信也科技杯-欺诈用户风险识别RobertAckley的初赛代码。 测评AUC最终为0.83631,排名33。本代码主要参照比赛baseline代码 https://github.com/DGraphXinye/2022_finvcup_baseline   
请在比赛网站上下载"初赛数据集.zip"文件，将zip文件中的"phase1_gdata.npz"放到路径'./xydata/raw'中。  


## Environments
Implementing environment:  
- python = 3.7.6
- numpy = 1.21.2  
- pytorch = 1.6.0  
- torch_geometric = 1.7.2  
- torch_scatter = 2.0.8  
- torch_sparse = 0.6.9
- networkx = 2.6.3
- pandas = 1.3.5
- scikit-learn = 1.0.2
- lightgbm = 3.3.2
- torchvision = 0.7.0
- tqdm = 4.64.0

详细见requirements.txt

- GPU: RTX A4000   


## Training

- **GraphSAGE (NeighborSampler)**
```bash
python train_mini_batch.py --model sage_neighsampler --epochs 200 --device 0
python inference_mini_batch.py --model sage_neighsampler --device 0
```

## Results:

| Methods   | Valid AUC  | Test AUC  |
|  :----  |  ---- | ---- |
| GraphSAGE (NeighborSampler) embedding + features + LightGBM  | 0.8518 | **0.8363** |

## 解决方案及算法方案

本次比赛团队的解决方案主要包括三步：基于GraphSAGE的节点Embedding（与baseline一致），手工加入时序等特征，通过LightGBM分类

1. 基于GraphSAGE的节点Embedding（与baseline一致）

   ​	基于baseline代码中GraphSAGE（NeighborSampler）的AUC最高，团队使用该网络对数据集中的节点进行embedding。网络训练与baseline中一致，修改的点为将17-128-2的GraphSAGE原模型修改为12-128-64-2的新模型，最后一层为64-2的线性层作为分类器。网络训练完成后，只取GraphSAGE的前两层，如此便将所有节点转换成了64维的向量。

   ​	值得注意的是，得到embedding时不再采用验证集（能使测试AUC上升0.003左右），因此inference_mini_batch.py中使用的是XYGraphP1_no_valid作为数据集类。

2. 手工加入时序等特征

   最终每个节点为201维的特征向量（由以下特征向量直接拼接得到），其中：

   - 点的embedding特征向量，64维
   - 节点本身的特征向量，17维
   - 节点的相邻节点的类别的数量，有4类节点，测试节点为未知节点，算作0类和1类各一个，共4维。考虑到数据集为单向边，因此将节点出边入边分别分开计算，因此总共4x2为8维。
   - 节点的边的类别，共有11类边，统计节点的边的不同类别的数量。同理出边入边分别分开统计，共11x2为22维。
   - 节点的随时间的边的数量。将边的时序特征均分为45份，统计改节点45个时间段边的数量。同理出边入边分别分开统计，共45x2为90维。

3. LightGBM分类

   ​	最终通过LightGBM分类器分类，参数设置具体见代码，1200个epoch的设置是由之前包含验证集的相同设置的实验推测而来。

## 复现流程

1. 首先通过python train_mini_batch.py --model sage_neighsampler --epochs 200 --device 0命名训练网络。
2. 网络Embedding，手工特征，LightGBM分类只需要运行python inference_mini_batch.py --model sage_neighsampler --device 0命令即可。



