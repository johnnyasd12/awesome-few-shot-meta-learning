Awesome Few-shot/Meta Learning Papers
===

[TOC]

# Few-shot with Domain shift

## One Shot Domain Adaptation for Person Re-Identification. 2018

## Meta-Learning with Domain Adaptation for Few-Shot Learning under Domain Shift, ICLR 19 rejected
- ProtoNet + CycleGAN?
### reviewer comment at [OpenReview](https://openreview.net/forum?id=ByGOuo0cYm)
- The proposed approach consists of **combining** a known few shot learning model, prototypical nets, together with image to image translation via CycleGAN for domain adaptation.  Thus the **algorithmic novelty is minor** and amounts to combining two techniques to address a different problem statement. 
- though meta learning could be a solution to learn with few examples, the solution being used in this work is **not meta learning** and so should not be in the title to avoid confusion.

## Few-shot adversarial domain adaptation. NIPS 2017
- [中文](https://blog.csdn.net/Adupanfei/article/details/85164925)
- [中文2](https://www.twblogs.net/a/5c1f39d2bd9eee16b3da81c0)
- [code (PyTorch)](https://github.com/Coolnesss/fada-pytorch)
- [My paper note](https://hackmd.io/8H_J9XauQgWrGfkLz88dKQ?view)
- supervised domain adaptation
- 並不真的 focus 在 few-shot learning
- 先在 source target 上 pre-train 一個 variational auto-encoder(VAE)，複製給 target task。兩個 task share 一些 layer，target task 只能 update task-specific layer；source task 可以 update shared 跟 他自己的 task-specific layer

## Transferable meta learning across domains. UAI 2018
- 似乎用到 target domain 的 unlabeled data
- 這篇也跟樓上一樣不是真的在做 few-shot
- [my paper note](https://hackmd.io/yh6uPnEwQzOfuvYOynB06Q)

### Abstract
- Meta-learning algorithms require **sufficient tasks** for meta model training and resulted model can **only solve new similar tasks**. 
- to address these two problems, we propose a new **transferable meta learning (TML)** algorithm


## Learning Embedding Adaptation for Few-Shot Learning. arXiv 1812
- [code - official (PyTorch)](https://github.com/Sha-Lab/FEAT)
- transformer + ProtoNets?
- SOTA: EA-FSL / FEAT
- 似乎沒用到 target task 的 data
- [my paper note](https://hackmd.io/DIpAhhCjQuSErytQoTkUog)

## Domain adaption in one-shot learning. ECML-PKDD 2018

- title 的字並沒有打錯ㄛ
- 似乎需要 target domain labeled data
- [My paper note](https://hackmd.io/VqkHvFGIToqGWsfSwLg6bA?view)
- [code - official (TF)](https://github.com/leonndong/DAOSL)
### Abstract
- given only one example of each new class. Can we **transfer knowledge learned by oneshot learning from one domain to another**?
- propose a **domain adaption framework based on adversarial networks**. 
- This framework is **generalized for situations where the source and target domain have different labels**.
- use a **policy network**, inspired by human learning behaviors, to effectively **select samples from the source domain in the training process**. This sampling strategy can further improve the domain adaption performance.

## A Closer Look at Few-shot Classification. ICLR'19

- [中文](https://zhuanlan.zhihu.com/p/64672817)
- [code - official (PyTorch)](https://github.com/wyharveychen/CloserLookFewShot)
- 提出兩個普通 baseline，發現許多情況可以和 SOTA 的 fewshot learning 媲美
- 比較的 SOTA 方法：MatchingNet、ProtoNet、RelationNet、MAML
- domain 差異小的情況下(例如CUBS)，隨著 baseNN 越強，不同 SOTA 方法的差異越小
- domain 差異大的情況下(例如miniImageNet)，隨著 baseNN 越強，不同 SOTA 方法的差異越大
- 有領域飄移情況發生時，SOTA 方法甚至沒有 baseline 表現好
- 特別強調 SOTA 在 domain adaptation 做得不好

### Reviewer Comment
- The conclusion from the network depth experiments is that “**gaps among different methods diminish as the backbone gets deeper**”. However, in a **5-shot mini-ImageNet case, this is not what the plot shows**. Quite the opposite: the gap increased. Did I misunderstand something? Could you please comment on that?
    - **跟我想問的問題一樣**
    - Authors' Answer: Sorry for the confusion. As addressed in 4.3, gaps among different methods diminish as the backbone gets deeper *in the CUB dataset*. In the mini-ImageNet dataset, the results are more complicated due to the domain difference. We further discuss this phenomenon in Section 4.4 and 4.5. We have clarified related texts in the revised paper. 

## Few-shot Learning with Meta Metric Learners. arXiv 1901.09890
- Microsoft AI & Research, IBM Research AI, JD AI Research
- Sentence Classification Services / Omniglot / Amazon Reviews

### Abstract
- Existing meta-learning or metric-learning based few-shot learning approaches are **limited in handling diverse domains** with various number of labels. 
### Conclusion
- we proposed a meta metric learner for few-shot learning, which is a **combination of an LSTM meta-learner and a base metric classifier**.
- The proposed method takes several advantages such as is able to **handle unbalanced classes** as well as to **generate task-specific metrics**.



## Subspace Networks for Few-shot Classification. arXiv 1905.13613


# 優先

## Adaptive Cross-Modal Few-Shot Learning, ICLR 2019 Workshop LLD, arXiv 1902
- 這篇跟我本來想到的 idea 一樣，居然有人做過了，機車
- 不過也許還沒做 domain shift?
- 對，他沒做 domain shift 哈哈哈哈

### Reviewer Comment
- this paper is not really doing few-shot learning, because according to section 3.2. and the experiments, the authors use the test labels in order to know which word embeddings to assign to each sample: "[...] containing label embeddings of all categories in D_train ∪ D_test". In other words, the authors use the labels (which are the goal of the classification task) to find the match between the two input modalities (to know what Glove vector to assign to each image).
- the experiments compare the results only between this multimodal approach and visual approaches. I believe using the Glove embeddings alone (no visual input) could give very good results on their own, and it is thus crucial for the authors to compare with this scenario too.
- the explanation for why you chose this form for lambda_c is unclear: "A very structured semantic space is a good choice for conditioning." 

## Label efficient learning of transferable representations across domains and tasks. NIPS 2017
- initialize the CNN for the target tasks in the target domain by a **pre-trained CNN learning from source tasks** in source domain. During training, they use an **adversarial loss** calculated from **representations in multiple layers** of CNN to force the two CNNs projects samples to a **task-invariant space**.

## Fine-grained visual categorization using meta-learning optimization with sample selection of auxiliary data. ECCV 2018
- done by **sharing the first several layers** of two networks to learn the generic information, while **learning a different last layer** to deal with different output for each task.


## Task Agnostic Meta-Learning for Few-Shot Learning. CVPR’19
- https://hackmd.io/2e6l5BmYS2ebhE__dwOcgQ?both#Task-Agnostic-Meta-Learning-for-Few-Shot-Learning-CVPR%E2%80%99191

## Delta-encoder: an effective sample synthesis method for few-shot object recognition. NIPS 2018
- Data method: learned transformation

### Abstract
- Our approach is based on a **modified auto-encoder**, denoted delta-encoder, that learns to **synthesize new samples for an unseen category just by seeing few examples** from it. The synthesized samples are then used to train a classifier.
- proposed approach learns to both **extract transferable intra-class deformations**, or "**deltas**", between same-class pairs of training examples, and to **apply those deltas** to the few provided examples of a **novel class** (unseen during training) in order to efficiently **synthesize samples from that new class**.

### the delta-encoder
![](https://i.imgur.com/XCGtXRv.png)
![](https://i.imgur.com/rl5xJtP.png)
- The simple key idea of this work is to **change the meaning of $E(X)$** from representing the "essence" of $X$, to representing the delta, or **"additional information" needed to reconstruct $X$ from $Y$** (an observed example from the same category).
    - $E$ for encoder, $D$ for decoder

## Low-shot learning with large-scale diffusion. CVPR 2018
- Data method: transform other dataset

## One-shot Learning with Memory-Augmented Neural Networks. arXiv'16
- 这篇论文解释了单样本学习与元学习的关系


# Classic

## Siamese neural networks for one-shot image recognition. 2015

## Hypernetworks. ICLR 2017

## (SNAIL) A Simple Neural Attentive Meta-Learner. ICLR 2018
- episodic training
- [code (PyTorch)](https://github.com/eambutu/snail-pytorch)
- [code (PyTorch) - 2](https://github.com/sagelywizard/snail)
- [code (MXNet? Gluon)](https://github.com/seujung/SNAIL-gluon)
- [my paper note (unfinished)](https://hackmd.io/rYWjR821QpWFjWPdqZzCqw)


## Soft k-Means

## Meta-learning with memory-augmented neural networks. ICML 2016
- **最早用 external memory 解 FSL classification** 的
- [my paper note](https://hackmd.io/OuVnw8WuT7OAuttmNtFtvg)

### Abstract
- Architectures with **augmented memory** capacities, such as Neural Turing Machines (NTMs), offer the ability to **quickly encode and retrieve new information**, and hence can potentially obviate the *downsides of conventional models*.
    - When new data is encountered, the **conventional models must inefficiently relearn their parameters** to adequately incorporate the new information without catastrophic interference.
- We also introduce a **new method for accessing an external memory that focuses on memory content**, unlike previous methods that additionally use memory location-based focusing mechanisms.

## Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. ICML 2017
![](https://i.imgur.com/vtN6Wso.png)
- 上圖演算法 **重要**，李老師教的版本有點簡化了
- [code - official? (TF)](https://github.com/cbfinn/maml)
- [code - PyTorch](https://github.com/dragen1860/MAML-Pytorch)
- [我的 NTU lecture 筆記](https://johnnyasd12.gitbooks.io/machine-learning-ntu/content/2019-meta-learning.html) [[edit](https://legacy.gitbook.com/book/johnnyasd12/machine-learning-ntu/edit#/edit/master/2019-meta-learning.md?_k=710c5j)]
- [中文1](https://zhuanlan.zhihu.com/p/57864886)
- [中文2](https://zhuanlan.zhihu.com/p/40417018)
    - 第一次 update 參數得到 $\theta_t'$ 時，使用 support set；而真正要更新 $\theta$ 時，是使用 query set 得到的 loss- episodic training
- 又稱 MAML
- 任何使用 gradient descent 的模型都適用本方法
- 尋找一個模型的 initialize parameter


## Reptile: A Scalable Meta-Learning Algorithm. 2018
- episodic training

## Matching Networks for One Shot Learning. NIPS'16
![](https://i.imgur.com/yjy0v5S.png)
- [Andrej Karpathy note](https://github.com/karpathy/paper-notes/blob/master/matching_networks.md)
- [我的筆記](https://hackmd.io/s3jGbRDmSTWXmZKIg_ExHQ)
- [code (TF)](https://github.com/AntreasAntoniou/MatchingNetworks)
- [code (PyTorch)](https://github.com/BoyuanJiang/matching-networks-pytorch)
- [code (TF) - 2](https://github.com/markdtw/matching-networks)
- [code (Keras)](https://github.com/cnichkawde/MatchingNetwork)
- **提出 episodic training**
- attention、memory network(multi-hopping)
- 和 Siamese Network 不同的是：**Siamese Network 只學習一個 distance(或 similarity function)；而 Matching Network 直接 end-to-end 學習一個 nearest neighbor classifier**
- 使用 cosine similarity 作為 metric?



## Prototypical Networks for Few-shot Learning. NIPS'17

- [code - official (PyTorch)](https://github.com/jakesnell/prototypical-networks)
- [code - official? (PyTorch)](https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch)
- [code (PyTorch)](https://github.com/cyvius96/prototypical-network-pytorch)
- episodic training
- [my paper note](https://hackmd.io/Oc2fQyxCS-SdGMpu3U3QRw)

## Optimization as a model for fewshot learning. ICLR 2017
- LSTM-based meta learning

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
- episodic training
- 將 support set 和 query 的 embedding 做 concat，然後用 NN 計算相似程度。
- 同樣的 architecture **也可以用來做 ZSL**，只要把 support set 換成 class semantic vector 即可


## Dynamic few-shot visual learning without forgetting. CVPR'18
- SOTA
- reduce intra-class variance 的重要性

## A. Rapid adaptation with conditionally shifted neurons. ICML 2018
- SOTA: AdaResNet


# recommended from other lab

## NPRF: A Neural Pseudo Relevance Feedback Framework for Ad-hoc Information Retrieval. EMNLP 2018
- [code - official (TF+Keras)](https://github.com/ucasir/NPRF)
- support set 跟 query set 都跟 training set 的 data 算 KL-divergence，然後看 query set 跟 support set 的哪個 data 最像

# others

## Learning Classifiers for Target Domain with Limited or No Labels. ICML 19
- 好像不是做 domain shift 的 @@
- 分別對 few-shot learning, generalized zero-shot learning, domain adaptation 做了實驗

### Abstract
- We propose a novel visual attribute encoding method that encodes each image as a **low-dimensional probability vector** composed of **prototypical part-type probabilities**. 
- At **test-time** we **freeze the encoder and only learn/adapt the classifier** component to limited annotated labels in FSL; new semantic attributes in ZSL.


## One-Shot Unsupervised Cross Domain Translation. NeurIPS 2018
- few-shot + GAN
- They learn **separate embedding for source and target tasks in different domains** to map them into a task-invariant space, then learn a **shared classifier** to classify samples from all tasks.



## Meta-Learning with Latent Embedding Optimization. ICLR'19
![](https://i.imgur.com/g2oJbf2.png)
- [code - official (TF)](https://github.com/deepmind/leo)
- SOTA: LEO
- 解決 MAML 不能很好的處理 high dim 的 data，即使 deeper network 也不好
- 用 (encoder+relation net) 對 data 做 latent code，然後 decode 出 w，再用 w 去算 loss 對 z 做 MAML，(**最後得到的 w' 跟 x 做 內積完 softmax??**
- OpenReview:
    - contributions 有二：(1)本來 MAML 是固定 init params，現在他們把他變成低維 latent space。(2)依據 subproblem 的 input data 來決定 init params
## Meta-Learning Probabilistic Inference for Prediction. ICLR'19
- [code](https://github.com/Gordonjo/versa)

## Meta-Learning for Semi-Supervised Few-Shot Classification. ICLR 2018
- 概念跟樓下那篇似乎有點像

## Learning to Propagate Labels: Transductive Propagation Network for Few-shot Learning. ICLR’19
- 又稱 **TPN**
- [中文](https://blog.csdn.net/choose_c/article/details/86560074)
- [code - official (TF)](https://github.com/csyanbin/TPN)
- [code - official (PyTorch)](https://github.com/csyanbin/TPN-pytorch)
- 有點像是結合 meta-learning 跟 semi-supervised learning
    - **將 support set 的 label 在 query set 中傳遞**
- [My Paper Note(待補完)](https://hackmd.io/HlArvrYZS4S6ksipcvJtaQ)



## Adaptive Posterior Learning: few-shot learning with a surprise-based memory module. ICLR'19
- [中文](https://zhuanlan.zhihu.com/p/67388319)
- [code - official (PyTorch)](https://github.com/cogentlabs/apl)
- 只在一些 $(x^{(i)}, y^{(i)})$ 的 prediction loss 高於 threshold 時去更新 memory。因此相較於 differentiable memory(**_???_**)，計算成本降低了
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

## Meta-learning with differentiable closed-form solvers. ICLR'19
- ridge regression


## Meta-Learning For Stochastic Gradient MCMC. ICLR'19

## Unsupervised Learning via Meta-Learning. ICLR'19

## How to train your MAML. ICLR'19

# CVPR 2019

## Meta-Transfer Learning for Few-Shot Learning. CVPR'19
- [code(TF)](https://github.com/y2l/meta-transfer-learning-tensorflow)
- [中文](https://zhuanlan.zhihu.com/p/57134284) (僅佔小部份篇幅)
- [作者blog](https://blog.yyliu.net/meta-transfer-learning/)
- 方法 (**改良 MAML**)
    1. 用 training set 所有 sample 訓練一個 NN 來 classify，得到一個 feature extractor
    2. sample 多個 task，用 MAML 根據 support set 更新 classifier(最後一層?) 參數，再利用 query set 計算 gradient，微調 initial parameter
- **Hard Task(HT) meta-batch**
    - 故意選一些 fail 的 task 並重組那些 data 成為 harder tasks 以用來 adverse re-training，希望 meta-learner 能在困難中成長 0.0



## Transferrable Prototypical Networks for Unsupervised Domain Adaptation. CVPR'19

- 單純做 domain adaptation 而不是 few-shot，只是方法借用 few-shot 的 ProtoNet

## Adversarial Meta-Adaptation Network for Blending-target Domain Adaptation. CVPR'19
- 提出了新的 scenario: Blending-target Domain Adaptation (**BTDA**)
- 提出了 Adversarial Meta-Adaptation Network (**AMEAN**)

## Instance-Level Meta Normalization. CVPR'19

## Learning to Learn How to Learn: Self-Adaptive Visual Navigation using Meta-Learning. CVPR'19
- [code - official (PyTorch)](https://github.com/allenai/savn)
- 用 meta-learning 來學 Visual Navigation task
## IMAGE DEFORMATION META-NETWORK FOR ONE-SHOT LEARNING. CVPR'19

## Meta-Learning with Differentiable Convex Optimization. CVPR'19

## Task Agnostic Meta-Learning for Few-Shot Learning. CVPR'19
### 想讀

## AutoAugment: Learning Augmentation Strategies from Data. CVPR'19

## Divide and Conquer the Embedding Space for Metric Learning. CVPR'19

## Finding Task-Relevant Features for Few-Shot Learning by Category Traversal. CVPR'19

## Few-Shot Learning via Saliency-guided Hallucination of Samples. CVPR'19

## RepMet: Representative-based metric learning for classification and few-shot object detection. CVPR'19

## Spot and Learn: A Maximum-Entropy Image Patch Sampler for Few-Shot Classification. CVPR'19

## LaSO: Label-Set Operations networks for multi-label few-shot learning. CVPR'19

## Few-Shot Learning with Localization in Realistic Settings. CVPR'19

## Few Shot Adaptive Faster R-CNN. CVPR'19	

## Large-Scale Few-Shot Learning: Knowledge Transfer with Class Hierarchy. CVPR'19

## Revisiting Local Descriptor based Image-to-Class Measure for Few-shot Learning. CVPR'19

## Generalized Zero- and Few-Shot Learning via Aligned Variational Autoencoders. CVPR'19
- [code (PyTorch)](https://github.com/chichilicious/Generalized-Zero-Shot-Learning-via-Aligned-Variational-Autoencoders)
### Abstract
-  In this work, we take feature generation one step further and propose a model where a shared latent space of image features and class embeddings is learned by modality-specific aligned variational autoencoders. 


## SPNet: Semantic Projection Network for Zero-Label and Few-Label Semantic Segmentation. CVPR'19

## Dense Classification and Implanting for Few-shot Learning. CVPR'19

## Coloring With Limited Data: Few-Shot Colorization via Memory Augmented Networks. CVPR'19

## Variational Prototyping-Encoder: One-Shot Learning with Prototypical Images. CVPR'19

# ICML 2019

## TapNet: Neural Network Augmented with Task-Adaptive Projection for Few-Shot Learning. ICML'19
- 不是 domain shift
### Abstract
- Handling previously unseen tasks after given only a few training examples continues to be a tough challenge in machine learning. We propose TapNets, neural networks augmented with task-adaptive projection for improved few-shot learning. Here, employing a meta-learning strategy with **episode-based training**, **a network** and **a set of per-class reference vectors** are learned across widely varying tasks. At the same time, for every episode, **features** in the embedding space are **linearly projected** into a new space as a form of quick **task-specific conditioning**. The training loss is obtained based on a **distance metric between the query and the reference vectors** in the projection space. Excellent generalization results in this way. When tested on the Omniglot, miniImageNet and **tieredImageNet** datasets, we obtain state of the art classification accuracies under **various few-shot scenarios**.

## LGM-Net: Learning to Generate Matching Networks for Few-Shot Learning. ICML'19

## Infinite Mixture Prototypes for Few-shot Learning. ICML'19

## Taming MAML: Efficient unbiased meta-reinforcement learning. ICML'19

## Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables. ICML'19

## Sever: A Robust Meta-Algorithm for Stochastic Optimization. ICML'19

## Fast Context Adaptation via Meta-Learning. ICML'19

## Probable Guarantees for Gradient-Based Meta-Learning. ICML'19

## Meta-Learning Neural Bloom Filters. ICML'19

## Hierarchically Structured Meta-learning. ICML'19

## Online Meta-Learning. ICML'19

# ECCV 2019?

# ICCV 2019?

# AAAI 2019



###### tags: `fewshot learning`


