Awesome Few-shot/Meta Learning Papers
===

[TOC]

# Classic

## Hypernetwork

## SNAIL
- A Simple Neural Attentive Meta-Learner. 
- [code (PyTorch)](https://github.com/eambutu/snail-pytorch)
- [code (PyTorch) - 2](https://github.com/sagelywizard/snail)
- [code (MXNet? Gluon)](https://github.com/seujung/SNAIL-gluon)


## Soft k-Means

## Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. ICML 2017
![](https://i.imgur.com/vtN6Wso.png)
- 上圖演算法 **重要**，李老師教的版本有點簡化了
- 又稱 MAML
- 任何使用 gradient descent 的模型都適用本方法
- 尋找一個模型的 initialize parameter
- [我的 NTU lecture 筆記](https://johnnyasd12.gitbooks.io/machine-learning-ntu/content/2019-meta-learning.html) [[edit](https://legacy.gitbook.com/book/johnnyasd12/machine-learning-ntu/edit#/edit/master/2019-meta-learning.md?_k=710c5j)]
- [中文1](https://zhuanlan.zhihu.com/p/57864886)
- [中文2](https://zhuanlan.zhihu.com/p/40417018)
    - 第一次 update 參數得到 $\theta_t'$ 時，使用 support set；而真正要更新 $\theta$ 時，是使用 query set 得到的 loss

## Reptile: A Scalable Meta-Learning Algorithm. 2018


## Matching Networks for One Shot Learning. NIPS'16
![](https://i.imgur.com/yjy0v5S.png)
- [Andrej Karpathy note](https://github.com/karpathy/paper-notes/blob/master/matching_networks.md)
- [我的筆記](https://hackmd.io/s3jGbRDmSTWXmZKIg_ExHQ)
- [code (TF)](https://github.com/AntreasAntoniou/MatchingNetworks)
- [code (PyTorch)](https://github.com/BoyuanJiang/matching-networks-pytorch)
- [code (TF) - 2](https://github.com/markdtw/matching-networks)
- [code (Keras)](https://github.com/cnichkawde/MatchingNetwork)
- attention、memory network(multi-hopping)
- 和 Siamese Network 不同的是：**Siamese Network 只學習一個 distance(或 similarity function)；而 Matching Network 直接 end-to-end 學習一個 nearest neighbor classifier**
- 使用 cosine similarity 作為 metric?

## One-shot Learning with Memory-Augmented Neural Networks. arXiv'16
- 这篇论文解释了单样本学习与元学习的关系

## Prototypical Networks for Few-shot Learning. NIPS'17

- [code - official (PyTorch)](https://github.com/jakesnell/prototypical-networks)
- [code - official? (PyTorch)](https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch)
- [code (PyTorch)](https://github.com/cyvius96/prototypical-network-pytorch)

![](https://i.imgur.com/vMhqPny.png)
- $D$ 是整個 trainin set；$D_k$ 是整個 training set 中的 class $k$ data
- $f_\phi$ 是 embedding function
- Update loss 那項 本來可能是 $\log(\exp(d(f_\phi(x), c_k)))$，化簡後得到 $d(f_\phi(x), c_k)$

![](https://i.imgur.com/ExO57HS.png)
- support set 裡面同 class 的 image 的 embedding 的 mean 被稱作 prototype
- 訓練使得 query set 到自己類別 prototype 的距離越近越好；到其他類別 prototype 距離越遠越好
- 有做 additional 實驗，training task 用的 way 比 testing task 更多，效果比較好
- 使用 Bregman divergence 的 Euclidean distance 作為距離?
- 也可做 zero-shot learning

## TADAM: Task dependent adaptive metric for improved few-shot learning. NIPS 2018
- [code - official (TF)](https://github.com/ElementAI/TADAM)
- 借鑑 ProtoNet 思想
- metric scaling
    - 學習一個 scaling factor $\alpha$，這樣可更好的輸出 metric 大小在合適的範圍
- task conditioning
    - 利用 prototype 的平均值構造 task representation，然後利用 task representation 來改變 feature extractor 的 function，即具有 adaptation 的能力
- auxiliary task co-training
    - 也把所有 training data 用來訓練 feature extractor，做為輔助的 task 一起 train，能讓 feature 更 generalize

## Learning to Compare: Relation Network for Few-Shot Learning. CVPR'18
![](https://i.imgur.com/6IyzPBW.png)
- [中文](https://zhuanlan.zhihu.com/p/35379027)
- [code - official (PyTorch fewshot)](https://github.com/floodsung/LearningToCompare_FSL)
- [code - official (PyTorch zeroshot)](https://github.com/lzrobots/LearningToCompare_ZSL)
- [code (PyTorch)](https://github.com/dragen1860/LearningToCompare-Pytorch)
- 將 support set 和 query 的 embedding 做 concat，然後用 NN 計算相似程度。
- 同樣的 architecture **也可以用來做 ZSL**，只要把 support set 換成 class semantic vector 即可

## Few-shot adversarial domain adaptation. NIPS 2017
- [中文](https://blog.csdn.net/Adupanfei/article/details/85164925)
- [code (PyTorch)](https://github.com/Coolnesss/fada-pytorch)
- supervised domain adaptation

## Domain adaption in one-shot learning. ECML-PKDD 2018

- title 的字並沒有打錯ㄛ
- [My paper note](https://hackmd.io/VqkHvFGIToqGWsfSwLg6bA?view)


## One-Shot Unsupervised Cross Domain Translation. NeurIPS 2018
- few-shot + GAN

## One Shot Domain Adaptation for Person Re-Identification. 2018

## Few-shot adversarial domain adaptation
- [中文](https://www.twblogs.net/a/5c1f39d2bd9eee16b3da81c0)

# recommended from other lab

## Learning Embedding Adaptation for Few-Shot Learning
- [code - official (PyTorch)](https://github.com/Sha-Lab/FEAT)
- transformer + ProtoNets?

## NPRF: A Neural Pseudo Relevance Feedback Framework for Ad-hoc Information Retrieval. EMNLP 2018
- [code - official (TF+Keras)](https://github.com/ucasir/NPRF)
- support set 跟 query set 都跟 training set 的 data 算 KL-divergence，然後看 query set 跟 support set 的哪個 data 最像

# ICLR 2019

## Meta-Transfer Learning for Few-Shot Learning. CVPR'19
- [code(TF)](https://github.com/y2l/meta-transfer-learning-tensorflow)
- [中文](https://zhuanlan.zhihu.com/p/57134284) (僅佔小部份篇幅)
- [作者blog](https://blog.yyliu.net/meta-transfer-learning/)
- 方法 (**改良 MAML**)
    1. 用 training set 所有 sample 訓練一個 NN 來 classify，得到一個 feature extractor
    2. sample 多個 task，用 MAML 根據 support set 更新 classifier(最後一層?) 參數，再利用 query set 計算 gradient，微調 initial parameter
- **Hard Task(HT) meta-batch**
    - 故意選一些 fail 的 task 並重組那些 data 成為 harder tasks 以用來 adverse re-training，希望 meta-learner 能在困難中成長 0.0

## A Closer Look at Few-shot Classification. ICLR'19

- [中文](https://zhuanlan.zhihu.com/p/64672817)
- [code - official (PyTorch)](https://github.com/wyharveychen/CloserLookFewShot)
- 提出兩個普通 baseline，發現許多情況可以和 SOTA 的 fewshot learning 媲美
- 比較的 SOTA 方法：MatchingNet、ProtoNet、RelationNet、MAML
- domain 差異小的情況下(例如CUBS)，隨著 baseNN 越強，不同 SOTA 方法的差異越小
- domain 差異大的情況下(例如miniImageNet)，隨著 baseNN 越強，不同 SOTA 方法的差異越大
- 有領域飄移情況發生時，SOTA 方法甚至沒有 baseline 表現好
- 特別強調 SOTA 在 domain adaptation 做得不好

## Meta-Learning with Latent Embedding Optimization
![](https://i.imgur.com/g2oJbf2.png)
- [code - official (TF)](https://github.com/deepmind/leo)
- 解決 MAML 不能很好的處理 high dim 的 data，即使 deeper network 也不好
- 用 (encoder+relation net) 對 data 做 latent code，然後 decode 出 w，再用 w 去算 loss 對 z 做 MAML，(**最後得到的 w' 跟 x 做 內積完 softmax??**
- OpenReview:
    - contributions 有二：(1)本來 MAML 是固定 init params，現在他們把他變成低維 latent space。(2)依據 subproblem 的 input data 來決定 init params
## Meta-Learning Probabilistic Inference for Prediction. ICLR'19
- [code](https://github.com/Gordonjo/versa)

## LEARNING TO PROPAGATE LABELS: TRANSDUCTIVE PROPAGATION NETWORK FOR FEW-SHOT LEARNING. ICLR'19
- 又稱 **TPN**
- [中文](https://blog.csdn.net/choose_c/article/details/86560074)
- [code - official (TF)](https://github.com/csyanbin/TPN)
- [code - official (PyTorch)](https://github.com/csyanbin/TPN-pytorch)
- 有點像是結合 meta-learning 跟 semi-supervised learning

## Adaptive Posterior Learning: few-shot learning with a surprise-based memory module
- [中文](https://zhuanlan.zhihu.com/p/67388319)
- [code - official (PyTorch)](https://github.com/cogentlabs/apl)
- approximates probability distributions by **remembering the most surprising observations**
- algorithm can perform as well as state of the art baselines
- main contributions:
    - surprise-based signal to write items to memory, **not needing to learn what to write**. So easier and faster to train, and minimizes how much data stored
    - ***(不懂???) An integrated external and working memory architecture which can take advantage of the best of both worlds: scalability and sparse access provided by the working memory; and all-to-all attention and reasoning provided by a relational reasoning module.***
    - A training setup which steers the system towards learning an algorithm which **approximates the posterior without backpropagating through the whole sequence of data in an episode**.
- Conclusion
    - We introduced a self-contained system which can **learn to approximate a probability distribution with as little data and as quickly as it can**. This is achieved by:
        - putting together the training setup which encourages adaptation
        - an external memory which allows the system to recall past events
        - a writing system to adapt the memory to uncertain situations
        - a working memory architecture which can efficiently compare items retrieved from memory to produce new predictions
    - We showed that the model can
        - Reach **state of the art accuracy with a smaller memory footprint** than other meta-learning models by efficiently choosing which data points to remember.
        - **Scale to very large problem sizes** thanks to the use of an external memory module **with sparse access**.
        - ***(不懂???) Perform fewer than 1-shot generalization thanks to relational reasoning across neighbors.***

## Meta-learning with differentiable closed-form solvers
- ridge regression


## Meta-Learning For Stochastic Gradient MCMC

## Unsupervised Learning via Meta-Learning. ICLR'19

# CVPR 2019

## Transferrable Prototypical Networks for Unsupervised Domain Adaptation. CVPR'19

- 單純做 domain adaptation 而不是 few-shot，只是方法借用 few-shot 的 ProtoNet

## Adversarial Meta-Adaptation Network for Blending-target Domain Adaptation. CVPR'19
- 提出了新的 scenario: Blending-target Domain Adaptation (**BTDA**)
- 提出了 Adversarial Meta-Adaptation Network (**AMEAN**)

## Instance-Level Meta Normalization. CVPR'19

## Learning to Learn How to Learn: Self-Adaptive Visual Navigation using Meta-Learning. CVPR'19
- [code - official (PyTorch)](https://github.com/allenai/savn)
- 
## IMAGE DEFORMATION META-NETWORK FOR ONE-SHOT LEARNING

## Meta-Learning with Differentiable Convex Optimization

## Task Agnostic Meta-Learning for Few-Shot Learning

## AutoAugment: Learning Augmentation Strategies from Data

## Divide and Conquer the Embedding Space for Metric Learning

## Finding Task-Relevant Features for Few-Shot Learning by Category Traversal

## Few-Shot Learning via Saliency-guided Hallucination of Samples	

## RepMet: Representative-based metric learning for classification and few-shot object detection

## Spot and Learn: A Maximum-Entropy Image Patch Sampler for Few-Shot Classification

## LaSO: Label-Set Operations networks for multi-label few-shot learning

## Few-Shot Learning with Localization in Realistic Settings	

## Few Shot Adaptive Faster R-CNN	

## Large-Scale Few-Shot Learning: Knowledge Transfer with Class Hierarchy

## Revisiting Local Descriptor based Image-to-Class Measure for Few-shot Learning

## Generalized Zero- and Few-Shot Learning via Aligned Variational Autoencoders

## SPNet: Semantic Projection Network for Zero-Label and Few-Label Semantic Segmentation

## Dense Classification and Implanting for Few-shot Learning	

## Coloring With Limited Data: Few-Shot Colorization via Memory Augmented Networks

## Variational Prototyping-Encoder: One-Shot Learning with Prototypical Images

# ICML 2019

## TapNet: Neural Network Augmented with Task-Adaptive Projection for Few-Shot Learning

## LGM-Net: Learning to Generate Matching Networks for Few-Shot Learning

## LGM-Net: Learning to Generate Matching Networks for Few-Shot Learning

## Infinite Mixture Prototypes for Few-shot Learning

## Taming MAML: Efficient unbiased meta-reinforcement learning

## Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables

## Sever: A Robust Meta-Algorithm for Stochastic Optimization

## Fast Context Adaptation via Meta-Learning

## Probable Guarantees for Gradient-Based Meta-Learning

## Meta-Learning Neural Bloom Filters

## Hierarchically Structured Meta-learning

## Online Meta-Learning

# AAAI 2019



###### tags: `fewshot learning`


