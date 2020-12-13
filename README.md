# Pretrain-Recsys
This is our Tensorflow implementation for our WSDM 2021 paper:

Bowen Hao, Jing Zhang, Hongzhi Yin, Cuiping Li, Hong Chen. Pre-Training Graph Neural Networks for Cold-Start Users and Items Representation.

Environment Requirement
The code has been tested running under Python 3.6.12. The required packages are as follows:

tensorflow == 1.14.0
numpy == 1.16.0
tqdm == 4.50.2
scipy == 1.5.2
torch == 1.3.1


Runing instructions:
We test our model on three datasets, namely MovieLens-1M. MOOCs and Last.fm, and each dataset contains a folder. Each folder contrains five GNNs models, in which three models are contained in the GeneralConv.py, and two other models are contained in the FastGCN folder and the FBNE folder. 

Before running our model, please download the oracle embedding files from https://pan.baidu.com/s/1XAR9thNt8PTHSvPQ8P2p-g (the password is hwj3), and put the oracle embedding files in the corresponding folders. For example, you can download the mooc oracle embedding file, and put it in the mooc folder.  

For the pre-traning process, run ' python GeneralConv.py ', which consists of three GNNs pre-training models (GraphSAGE, GAT and LightGCN) or ' python FastGCNConv.py ' in the FastGCN folder or ' python FBNEConv.py ' in the FBNE folder. For the downstream recommendation process, run ' python downstream_trainer.py ' in the basic folder or the FastGCN/FBNE folder. 
