A Survey on Few-Shot Learning
=

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

論文名稱：Generalizing from a Few Examples: A Survey on Few-Shot Learning. arXiv 2019

[TOC]

# 1 Introduction

2005、2006年即提出 one-shot learning[^oneshot06][^singleexp05] 的概念。
- 另一篇 survey 論文[^surveyFewshot]是關注 concept learning & experience learning for small sample，而本筆記的論文關注 few-shot learning

FSL(few-shot learning) 其中一個效用是能夠幫助我們減少一些 application 所需要的 data 數量，有關 FSL 的研究例如：
- image classification (Matching Networks[^matchNet])
- image retrieval ([^fewRetrieval])
- object tracking ([^fewTrack])
- gesture recognition 
- image captioning & visual question answering 
- video event detection
- language modeling
- architecture search

另外一個 FSL 的 scenario 是 supervised information 太難拿到，或者 target task 沒有太多 examples，有關 FSL 研究例如：
- drug discovery
- cold-start item recommendation
- FSL translation

以下是一些做 few shot learning 的方法
- meta learning 的方法
    - memory network[^memoNet]
- embedding 的方法
    - matching network[^matchNet]
- generative modeling 的方法
    - 0.0[^neuralStatistician]

本篇 survey 的貢獻：
- 形式化的定義了 few shot learning (FSL)，該定義足以包含所有存在的 FSL，且明確到足以闡明 FSL 的目標以及可以如何解決。
- 指出 FSL 基於 ML error decomposition[^errDecomp] 的問題。弄清楚是因為不可靠的 empirical risk minimizer 使得 FSL 難以學習，而這可以透過**簡化或者 satisfying(???) sample complexity** 來解決
- 從 FSL 出生到最新的都做了 review，並且對 data、model、algorithm 分類，評估優劣。
- 對於 problem setup、techniques、applications、theories 提出了 4 個**有前途的 FSL 未來方向**

# 2 Overview

## 2.1 Notation

(如果說 machine learning 是 deal with input ) 
- 對於一個 supervised learning task $T$, FSL deals with $D = \{D^{train}, D^{test}\}$，個人理解 $D$ 就是 few-shot learning 要處理的 **一筆資料**
- $D^{train} = \{(x^{(i)}, y^{(i)})\}_{i=1}^I$ where $$I$$ is small
- 一般會考慮 **$N$-way-$K$-shot classification task**
    - 即 $N$ classes，每個 class 有 $K$ examples
    - 此時 $I = KN$
    - (對於每個 task?) $p(x, y)$ 表示 ground truth joint distribution，而 $\hat h$ 表示 $x$ 映射到 $y$ 的最佳 hypothesis
    - model 決定了 hypothesis space $\mathcal H$
    - 預測值 $\hat y = h(x ; \theta)$

## 2.2 Problem Definition

*Definition 2.1. (**Machine Learning**)* A computer program is said to learn from experience $E$ with respect to some classes of task $T$ and performance measure $P$ if its performance can improve with $E$ on $T$ measured by $P$. 

*Definition 2.2. **Few-Shot Learning*** (FSL) is a type of machine learning problems (specified by $E$, $T$ and $P$) where $E$ contains a little supervised information for the target $T$. 

## 2.3 Relevant Learning Problems
- Semi-supervised learning
- Imbalanced learning
- Transfer learning
- Meta-learning
    - 許多 FSL 的方法都是 meta-learning methods，使用 meta-learner 當成 prior knowledge

## 2.4 Core Issue

以 error decomposition 說明 FSL 主要的問題。

### 2.4.1 Empirical Risk Minimization

### 2.4.2 Unreliable Empirical Risk Minimizer

### 2.4.3 Sample Complexity


## 2.5 Taxonomy
基於 prior knowledge 是如何被使用的，我們將目前的 FSL works 分成以下類別：
- *Data*: 用 prior knowledge 來擴充 $D^{train}$ 從 $I$ samples 變成 $\tilde I$ samples 
- *Model*: 基於 prior knowledge in experience $E$ 來限制 $\mathcal H$ 複雜度的方法
- *Algorithm*: 利用 prior knowledge 以提供最佳 hypothesis $h^*$ 的參數 $\theta$，提供很好的 initial point 以開始 search；又或者利用 prior knowledge 直接提供 search steps。

### 目前的 work 可以整合成以下分類
![](https://i.imgur.com/YeTkNNx.png)


# 3 Data method

![](https://i.imgur.com/AL0Ok1m.png)

## 3.1 Transform $D^{train}$

以下 paper list 待補
### 3.1.1 Handcrafted Rule
以 image recognition tasks 而言，許多 handcrafted rules 作為 preprocessing routine，例如：translating(???)、flipping、shearing、scaling、reflecting、cropping、rotating。

### 3.1.2 Learned Transformation
另外一種轉換 $D^{train}$ 的方式，自己 learn
- Delta-encoder: an effective sample synthesis method for few-shot object recognition. NIPS 2018

### 3.1.3 Discussion
few-shot learning 的設定中，$p(x, y)$ 的分布並不明顯，因此 handcrafted rules 並沒有考慮 $D^{train}$ 中 task 或 data 的特性；而 learned transformation 雖然可以利用 $D^{train}$ 的 knowledge，然而需要從非常相似的 task 萃取 prior knowledge，這代價可能很高，因此不總是可行。

## 3.2 Transform Other Data Sets

以下 paper list 待補
### 3.2.1 Weakly Labeled or Unlabeled Data Set
類似於 semi-supervised 的 label propagation 思路?
- Low-shot learning with large-scale diffusion. CVPR 2018

### 3.2.2 Similar Data Set
例如 老虎的 dataset 更像貓的 dataset，而 dataset 雖然 similar 但是 target FSL class 不一樣，因此有人提出使用 GAN 生成合成的 $\tilde x$ 來做 data aggregation
- Low-shot Learning via Covariance-Preserving Adversarial Augmentation Networks. NIPS 2018

### 3.2.3 Discussion
weakly labeled or unlabeled data
- 容易 access data
- quality 較低

similar dataset
- 何謂 similar ? 這較難決定
- quality 較高
## 3.3 Summary
- 第一種做法 (transform $D^{train}$) 產生的新 sample 不會離 $D^{train}$ 太遠，因此有其限制。
- 第二種做法 (transform from other data and adapt them to mimic $(x^{(i)}, y^{(i)})$)，可以有大的 variation，但是 adapt samples to be like $(x^{(i)}, y^{(i)})$ 是很困難的。
- 總之，因為 $p(x, y)$ 是未知的，不可能得到完美的 prior knowledge，這代表 augmentation procedure 並不精確，**estimated one(y?) 和 ground truth 之間的 gap 很大程度上影響了 data quality，甚至可能導致 concept drift**。

# 4 Model method

如果一般的 ML model 要用來處理 few-shot $D^{train}$，那它必須要選擇一個小的 hypothesis space $\mathcal H$，然而 real-world 的問題常常是非常複雜的，因此不能用一個小的 $\mathcal H$ 來表示。接下來會看到，在這個 section 中我們利用 $E$ 中的 prior knowledge 補足缺乏的 samples 來學習一個很大的 $\mathcal H$。特別的是，利用 prior knowledge 來影響 $\mathcal H$ 的選擇，像是限制 $\mathcal H$。用這樣的方法，就減少了 sample complexity，如此 empirical risk minimization 更可靠，overfitting 的風險就更小。依據**被使用的 prior knowledge 有所不同**，這些方法可以被分為以下四類：
1. multitask learning
2. embedding learning
3. learning with external memory
4. generative modeling

![](https://i.imgur.com/kEYSw1x.png)

## 4.1 Multitask Learning
相關的 task $T_t$ 包含了 few samples 跟 many samples 的 task，我們稱 few-shot task 為 *target tasks*；稱其他 task 為 *source tasks*。因為這些 task 相關，他們應該會有相似的或者重疊的 hypothesis space $\mathcal H_{T_t}$。這可以用 sharing parameters 做到，這樣可以對 $H_{T_t}$ 下 constraint。
依據 parameter sharing 是否明確的被 enforced，這種 strategy 可以被分為
1. hard parameter sharing
2. soft parameter sharing

![](https://i.imgur.com/U5my2Cp.png)

### 4.1.1 Hard Parameter Sharing

- 論文：Fine-grained visual categorization using meta-learning optimization with sample selection of auxiliary data. 2018, ECCV
    - 不同 task，share 前幾層；但是不 share 最後一層，因為不同 task 有不同 output
- 論文：One-Shot Unsupervised Cross Domain Translation. 2018, NeurIPS
    - 和上一篇相反，source & target task 學習不同的 embedding function，讓他們 map 到 task-invariant space，再學一個 shared classifier 來 classify 所有 task ***(可是 task output 不一樣要怎麼 share classifier?)***
    - few-shot + GAN
- 論文：Few-shot adversarial domain adaptation. 2017, NIPS
    - 先在 source target 上 pre-train 一個 variational auto-encoder(VAE)，複製給 target task。兩個 task share 一些 layer，target task 只能 update task-specific layer；source task 可以 update shared 跟 他自己的 task-specific layer

### 4.1.2 Soft Parameter Sharing
本類方法不強制 share parameter，而是鼓勵不同 task 的 params 相似，有相似的 $\mathcal H_{T_t}$

- 論文：Multi-Task Transfer Methods to Improve One-Shot Learning for Multimedia Event Detection. 2015, BMVC
    - 對所有 task 的 $\theta_{T_t}$ 的所有 combination 做 pairwise penalty，希望他們(參數?)越像越好

- 另一種方法是經由 loss 來調整 $\theta_{T_t}$，optimization 之後，學到的 $\theta_{T_t}$ 也會利用到彼此的資訊 ***(???)***

### 4.1.3 Discussion

- 本類方法利用 explicitly 或 implicitly 共享參數來限制 $H_{T_t}$。
- 以 soft parameter sharing 的方法而言，如何施加 similarity constraint 需要良好的設計。

## 4.2 Embedding Learning

*Embedding learning* methods 將 $x^{(i)}\in\mathcal X \subseteq\mathbb R^d$ embed 成更小的 embedding space $z^{(i)}\in\mathbb Z\subseteq\mathbb R^m$ 使得相似與不相似的 pari 非常容易被分辨，因此 $\mathcal H$ 是被限制的。embedding function 主要經由 prior knowledge 學習，且可以**額外利用 $D^{train}$ 帶來 task-specific information**。注意 embedding learning methods 主要是為 classification task 所設計。

![](https://i.imgur.com/cS2ssL8.png)
**key components (可參考上圖):**
- function $f(\cdot)$ 將 $x^{test}\in D^{test}$ embed 到 $\mathcal Z$
- function $g(\cdot)$ 將 $x^{(i)}\in D^{train}$ embed 到 $\mathcal Z$
- $s(\cdot , \cdot)$ 計算 $f(x^{test})$ 和 $g(x^{(i)})$ 的相似度
- 於是 $x^{test}$ 就被 assign 給最像的 $x^{(i)}$ 的 class
- $f$ 和 $g$ 可相同，**可不同**，因為 $x^{test}$ 可以利用 $D^{train}$ 的資訊來做 embedding 以調整 **comparing interest(???)**。例如
    - Learning feed-forward one-shot learners. NIPS 2016
    - Matching networks for one shot learning. NIPS 2016

![](https://i.imgur.com/NBiNZgx.png)

### 4.2.1 Task-specific
本方法學習一個為 $D$ 量身訂做的 embedding function
- Few-shot learning through an information retrieval lens. 2017
    - 這篇 paper 學習一個可以保持 $(x^{(i)}, y^{(i)})\in D^{train}$ ranking list 的 embedding，**ranking list 表示同一個 class 的 rank 較高，不同 class 的 rank 較低**
### 4.2.2 Task-invariant
從一個不包含 $D$ 的 large set of dataset $D_c$ 學習一個 embedding function。且不需要 re-training
- Object classification from a single example utilizing class relevance metrics. 2005, NIPS
    - 提出第一個 few-shot learning 的 embedding method
- Siamese neural networks for one-shot image recognition. 2015
    - 懂
### 4.2.3 A Combination of Task-invariant and Task-specific
1. *Learnet*
2. *Matching Nets*
3. *ProtoNet*
4. *Relative representations*
5. *Relation graph*
6. *SNAIL*
#### Learnet
- 論文：*Learning feed-forward one-shot learners. NIPS 2016*
- 給定 task，預測參數
- map $x^{(i)}$ 到每一層 conv siamese net 的 parameter

DyConvNet
- 論文：Dynamic Conditional Networks for Few-Shot Learning. ECCV 2018
- 為了減少 learner 的參數量，DyConvNet 使用一個固定的 filter set，learner 只學習如何從 filter set 選出 filter 來組合

Recent Work
- Meta-learning with differentiable closed-form solvers. ICLR 2019
- 將 *Learnet* 的 classification layer 替換成 ridge regression model，如此它的 parameter can be found by cheap closed-form solution。

#### Matching Nets
_Matching Network_
- 論文：Matching networks for one shot learning. NIPS 2016
- [我的筆記](https://hackmd.io/2e6l5BmYS2ebhE__dwOcgQ?view#Matching-Networks-for-One-Shot-Learning-NIPS%E2%80%9916)

然而 matching nets 會導致相鄰的 example 對彼此有更大的影響力。為了去除這樣的順序關係，Altae-Tran 等人將 biLSTM 換成 LSTM+attention，並迭代精煉 $g$ 和 $f$ 來 encode contextual information。
- 論文：Low Data Drug Discovery with One-Shot Learning. 2017

_active learning 的變體_
- 論文：Learning Algorithms for Active Learning. ICML 2017
- 在 matching nets 上加了 sample selection stage

#### ProtoNet
_ProtoNet_
- Prototypical Networks for Few-shot Learning. NIPS 2017
- [我的筆記](https://hackmd.io/2e6l5BmYS2ebhE__dwOcgQ?both#Prototypical-Networks-for-Few-shot-Learning-NIPS%E2%80%9917)
- 只在 $x^{test}\in D^{test}$ 和每個 class 的 prototype 使用一次的比較
- class $n$ 的 prototype 被定義為 class 中 embedding 的平均值
    - prototype $c_n = \frac{1}{K}\sum_{k=1}^K g(x^{(i)})$ 而 $x^{(i)}$ 是第 n 個 class 中的 K 個 example， $c$ for center
- ProtoNet 用同一個 CNN 對 $x^{(i)}$ 和 $x^{test}$ 做 embedding，忽略了不同 $D^{train}$ 的特性

結合 matching nets 和 ProtoNet 以納入(?) task-specific information
- Low-shot learning from imaginary data. CVPR 2018

_TADAM_
- TADAM: Task dependent adaptive metric for improved few-shot learning. NIPS 2018
- [我筆記](https://hackmd.io/2e6l5BmYS2ebhE__dwOcgQ?view#TADAM-Task-dependent-adaptive-metric-for-improved-few-shot-learning)
- 將所有 prototype $c_n$ 取平均當成 task embedding，然後 map 到 ProtoNet 所使用 CNN 的一些 parameters

_semi-supervised variant of ProtoNet_
- Meta-Learning for Semi-Supervised Few-Shot Classification. ICLR 2018

#### Relative representations
**ARC**
- 論文：Attentive Recurrent Comparators. 2017, ICML
- ARC 使用 RNN+attention 來循環比較 $x^{test}$ 的不同 region 跟每個 class 的 prototype $c_n$，並且產生 relative representation，外加使用 biLSTM 將其他的比較 embed 成 final embedding

**Relation Net**
- 論文：Learning to compare: Relation network for few-shot learning. CVPR 2018
- [我的筆記](https://hackmd.io/2e6l5BmYS2ebhE__dwOcgQ?both#Learning-to-Compare-Relation-Network-for-Few-Shot-Learning-CVPR%E2%80%9918)
- 首次將 $x^{test}$ 和 $x^{(i)}$ 一起 embed 到 space $\mathcal Z$，然後 concat 成 relative representation，再用另個 CNN output similarity score
#### Relation graph
是保留了所有 pairwise sample relationship 的 graph，$D^{train}$ 和 $D^{test} 的 sample 當 node，node 之間的 edges 被 $s$ 決定 ($s$ 是 learn 出來的)，於是 $X^{test}$ 就用 neighborhood information 來 predict。

用 GCN 來做 few-shot
- 論文：Few-Shot Learning with Graph Neural Networks. 2018, ICLR

把 $x^{(i)}$ 跟 $x^{test}$ embed 到 $\mathcal Z$ 並建造了 relation graph，且用 **closed-form label propagation rules** 將 $y^{test}$ label
- 論文：LEARNING TO PROPAGATE LABELS: TRANSDUCTIVE PROPAGATION NETWORK FOR FEW-SHOT LEARNING. ICLR 2019
- [我的筆記](https://hackmd.io/2e6l5BmYS2ebhE__dwOcgQ?both#LEARNING-TO-PROPAGATE-LABELS-TRANSDUCTIVE-PROPAGATION-NETWORK-FOR-FEW-SHOT-LEARNING-ICLR%E2%80%9919)
#### SNAIL
- 論文：A Simple Neural Attentive Meta-Learner. ICLR 2018
- 我的筆記?
- 交錯的 temporal convolution layers & attention layers
    - temporal convolution 用來結合過去 time steps 的 information
    - attention 選擇性的聚焦在「和目前的 input 有關聯性」的 time step


### 4.2.4 Discussion
Task-specific embedding 完整的考量了 $D$ 的 domain knowledge，然而，給定的 few-shot $D^{train}$ 可能是 biased 的，只從 $D$ 學可能不恰當。
原文：
> Task-specific embedding fully considers the domain knowledge of D. However, the given few-shot Dtrain can be biased, only learning from them may be inappropriate. Modeling ranking list among Dtrain has a high risk of overfitting to Dtrain. Besides, H learned this way cannot generalize from new tasks or be adapted easily. Using pre-trained task-invariant embeddings has a low computation cost. However, the learned embedding function does not consider any task-specific knowledge. When special is the reason that Dtrain has only a few examples such as learning for rare cases, simply applying task-invariant embedding function can be not suitable. A combination of task-invariant and task-specific information is usually learned by meta-learning methods. They can provide a good H and quickly generalize for different tasks by learner. However, how to generalize for a new but unrelated task without bringing in negative transfer is not sure.
## 4.3 Learning with External Memory
類似於 Neural Turing Machine(NTM) 和 memory networks
memory networks
- 論文：Neural turing machines. arXiv 2014
- 論文：End-to-end memory networks. 2015, NIPS
- 論文：Memory networks. arXiv 2014
- [My memory network & neural turing machine NTU 筆記](https://johnnyasd12.gitbooks.io/machine-learning-ntu/content/li-hong-yi-advanced-topics-in-deep-learning/conditional-generation-by-rnn-and-attention.html)

之前的(***某些 approach***) 給定一個新的 task $T$，model 就要 re-train 來合併 $D_{train}$ 的 information，非常 costly。然而 learning with external memory 直接記住需要的 knowledge 要被 retrieve 或者 update，因此減輕了 learning 的負擔，而且可以快速 generalize。

### Notations
- $M\in\mathbb R^{b\times m}$：memory 有 $b$ 個 memory slots，每個 memory slots 是 $m$ 維向量
- $M(i)\in\mathbb R^m$：第 i 個 memory slot
- $f$：embedding function
- $q = f(x^{(i)})\in\mathbb R^m$：query 即 $x^{(i)}$ 的 embedding

利用 similarity 來決定要從 memory 萃取哪個 knowledge，下圖為不同方法 with external memory 的特性整理
![](https://i.imgur.com/yXyZqKD.png)

對 FSL 而言，$D^{train}$ 只有少量 samples，re-train model 不太可行。
而 learning with external memory 可以解這個問題
- 它將 extracted knowledge 從 $D^{train}$ 存進 memory
- embedding function $f$ 沒被 re-train
- 因此 initial hypothesis space $\mathcal H$ 也沒變
- ***When a new sample comes, relevant contents are extracted from the memory and combine into the local approximation for this sample. Then the approximation is fed to the subsequent model for prediction, which is also pre-trained.(不懂***
- 因為 $D^{train}$ 存在 memory，所以 task-specific information 被有效的利用。

通常當 memory 沒滿的時候，新的 samples 可以被寫入空的 memory slots。然而當 memory 滿的時候，就必須用一些規則決定哪些 memory slots 要被更新或替換。目前的 work 可以被歸類為以下幾類：

### (1) Update the least recently used memory slot
**MANN**
- 論文：Meta-learning with memory-augmented neural networks. 2016, ICML
- 最早用 memory 解 FSL classification 的
### (2) Update by location-based addressing
在 Neural Turing Machine 中，總是 update 所有 memory。

**abstract memory** 就使用了上述 strategy
- 論文：Few-Shot Object Recognition from Machine-Labeled Web Images. 2017, CVPR
### (3) Update according to the age of memory slots
對 memory 紀錄每個 memory slot 的 age，當某個 memory slot 被讀取，就增加 age；當 slot 被 update，就重製 age 為 0。最老的就像是過時的 information(***為啥?? 被讀取很多次不是代表很有用嗎***)，Life-long memory 和 CMN 都在 memory 滿的時候將最老的 memory slot 更新。

然而，有時候在 old memory slot 中的 rare events 很重要。為解這個問題，life-long memory 特別喜歡 update 相同 classes 的 memory slots。(這裡不確定，原文：
> However, some times one value the rare events in old memory slots. To deal with it, life-long memory specially prefers to update memory slots of the same classes. As each class occupies comparative number of memory slots, rare classes are protected in a way.

**Life-long memory**
- Learning to remember rare events. 2017, ICLR

**CMN**
- Compound Memory Networks for Few-shot Video Classification. 2018, ECCV
### (4) Update the memory only when the loss is high

**surprised-based memory module**
- Adaptive Posterior Learning: few-shot learning with a surprise-based memory module. ICLR 2019
- 只在一些 $(x^{(i)}, y^{(i)})$ 的 prediction loss 高於 threshold 時去更新 memory。因此相較於 differentiable memory(**_???_**)，計算成本降低了
### (5) Use the memory as storage without updating
***不太懂，原文：***
> MetaNet [82] stores sample-level fast weights for $(x^{(i)}, y^{(i)})\in D^{train}$ in a memory, and conditions its embedding and classification by the extracted fast weights, so as to combine the generic and specific information. MetaNet repeatedly applies the fast weight to selected layers of a CNN. In contrast, Munkhdalai et al. [2018] learn fast weight to change the activation value of each neuron, which has a lower computation cost.

**MetaNet**
- Meta Networks. 2017, NIPS

**change activation**
- Rapid adaptation with conditionally shifted neurons. ICML 2018

### (6) Aggregate the new information into the most similar one

**MN-Net**
- Memory Matching Networks for One-Shot Image Recognition. 2018, CVPR
- 將新 sample 的 information merge 到最像的 memory slots
- 本文 memory 是用來精煉 $f(x^{(i)})$ 並且參數化一個 CNN，就像 Learnet 一樣，而不像  matching nets 是直接 predict $x^{test}$。
- 於是每個 $x^{test}$ 就用這個 conditional CNN 來做 embedding

### 4.3.1 Discussion

我們可以單純把 $D^{train}$ 存到 memory 來 adapt 新的 task，就能簡單的辦到 fast generalization。

另外，在設計 memory 更新及存取的規則時，可以合併像是 lifelong learning 或者減少 memory updates 的 rules。**(?)**

**然而，設計這些 rule 需要 human knowledge，目前的 work 並沒有一個完勝的 winner，如何自動設計或選擇 update rules 是重要的議題。**

## 4.4 Generative Modeling

暫時略過

### 4.4.1 Parts and Relations

### 4.4.2 Super Classes

### 4.4.3 Latent Variables

#### (1) Variational auto-encoder (VAE)

#### (2) Autoregressive model

#### (3) Inference network

#### (4) Generative adversarial networks (GAN)

#### (5) Generative version of Matching Nets (GMN)

### 4.4.4 Discussion

## 4.5 Summary

- **_Multitask learning_ 的方法隱式的擴充了資料，因為某些 parameter 用 multiple tasks 一起 learn。然而 target $D$ 必須是 $D_{T_t}$ 的其中之一，才能 joint training。因此每次有個新的 task，就要重新 train，成本高又很慢，因此不適合只有 one-shot 或希望 fast inference 的 task。**
- **_Embedding learning_ 的方法大多數一旦學習完，利用 forward pass 後對 embedded samples 做 nearest neighbor 就很容易 generalize 到新的 task。_然而，如何混和 invariant 和 specific 的資訊是不確定的(???)_** 後面原文：
    > However, how to mix the invariant and specific information of tasks within θ in a principled way is unclear.
- **_Learning with external memory_ 的方法利用 memory 精煉並重釋每個 sample，結果便 reshape 了 $\mathcal H$，且避免為了 $D^{train} 而 re-train，然而，本類方法需要更多的空間以及運算成本，因此，目前的 external memory size 有限制，故不能記住太多 information**
- **_Generative modeling_ 的方法學習一個 prior probability，塑造了 $\mathcal H$，有很好的解釋性、因果關係以及 compositionality(?)，他們可以透過 learning joint distribution $p(x, y)$ 處理更廣泛的任務類型，像是 generation 以及 reconstruction。學好的 generative models 可以做 data augmentation，然而本類方法計算成本很高，且難以和其他 model 比較(why??)。為了計算可行性，他們需要對結構做簡化而導致不精確的近似。**

# 5 Algorithm method

本 section 的方法並不限制 $\mathcal H$ 的 shape，因此一般 CNN 及 RNN 仍然可以使用。相對地，這些方法利用 prior knowledge 來改變搜尋 $\theta$ 的方式 $\theta^t = \theta^{t-1} - \alpha^t\nabla_{\theta^{t-1}}\ l^t(\theta^{t-1})$。

根據 prior knowledge 是如何影響 search strategy 的，我們將這些方法分成三類：
1. *Refine existing parameters* $\theta^0$
    - 從其他 task 學到的 initial $\theta^0$ 被用來初始化搜尋，然後由 $D^{train}$ 精煉 
    - 代表性的做法：一般的 fine-tune
2. *Refine meta-learned* $\theta$
    - meta-learner 是由 task distribution 抽出的不同 few-shot task 所訓練，輸出 general 的 $\theta$，然後每個 learner refines the $\theta$ provided by the meta-learner using $D^{train}$。
    - 代表性的研究: __MAML__
3. *Learn search steps*
    - 學一個 meta-learner 來輸出 search steps 或者 update rules，例如 update 的方向或者 step size

![](https://i.imgur.com/9JVHbEb.png)
## 5.1 Refine existing parameters $\theta^0$
這種 strategy pre-trained model 的 $\theta^0$ 當作一個很好的 initialization，然後用 $D^{train} 來 adapt。該 strategy 假設 $\theta^0$ 捕捉到非常 general 的 structures

### 5.1.1 Fine-tune $\theta^0$ with Regularization

- 實務上常常使用這種方法
- 先在 large-scale 的 data 例如 ImageNet 上 train 完再 adapt 到 smaller data sets
- 然而單純 fine-tune 很容易導至 overfitting，
![](https://i.imgur.com/rncyp06.png)

1. Early-stopping
    - 論文：Neural voice cloning with a few samples. NeurIPS 2018
        - 然而這種方法需要從 $D^{train}$ 分出 validation set 來監控 training 過程，使得 sample 數量又更少了。何況，使用很小的 validation set 會使得 searching strategy 有很高的偏差


2. Selectively update $\theta^0$
    - 此類方法只更新部分參數來防止 overfitting
    - 論文：Learning structure and strength of CNN filters for small sample size training. CVPR 2018
        - 使用一組固定的 filter set，只用 $D^{train}$ 來學習如何控制這些 filter 內的 ***multitude(?)*** of elements
    - 以下兩篇論文都直接對 $D^{train}$ 中的所有 class 加入 weight 成為新的columns，而 pre-trained weight 保持不變
        - 論文：Low-Shot Learning With Imprinted Weights. CVPR 2018
        - 論文：Few-shot image recognition by predicting parameters from activations. CVPR 2018

3. Cluster $\theta^0$
    - 把 parameter 分 group 之後，再對相同 group 使用相同的 update information，如此可以對 search strategy 做很強的限制
    - 論文：Efficient k-shot learning with regularized deep networks. 2018, AAAI
        - 他們使用輔助資料來對 pre-train model 的 filter 分 group，然後用 $D^{train}$ 來做 group-wise back-propagation

4. Model regression networks
    - 論文：Model regression networks for easy small sample learning. 2016, ECCV
    - 論文：CLEAR: Cumulative LEARning for One-Shot One-Class Image Recognition. CVPR 2018
    - 這裡看嚨無，原文：
    > Model regression networks [132] assume there exists a task-agnostic transformation from parameter trained using a few examples to parameter trained using many samples. Wang and Hebert [2016b] then refine θ0 learned with fixed N-way-K-shot problem. Similarly, Kozerawski and Turk [2018] learn to transform the embedding of x(i) to a classification decision boundary.
### 5.1.2 Aggregate a Set of $\theta^0$'s

暫時略過

1. Similar data set
2. Unlabeled data set

### 5.1.3 Fine-tune $\theta^0$ with New Parameters

暫時略過

### 5.1.4 Discussion

> Methods discussed in this section reduce the effort of doing architecture search for H from the scratch. Since directly fine-tuning can easily overfit, methods that fine-tune a θ0 with regularization turn to regularize or modify existing parameters. They usually consider a single θ0 of some deep model. However, suitable existing parameters are not always easy to find. Another way is to aggregate a set of parameters θ0’s from related tasks into a suitable initialization. However, one must make sure that the knowledge embedded in these existing parameters is useful to the current task. Besides, it is costly to search over a large set of existing parameters to find the relevant ones. Fine-tune θ0 with new parameters leads to more flexibility. However, given the few-shot Dtrain, one can only add limited parameters, otherwise overfitting may occur.

## 5.2 Refine Meta-learned $\theta$

**接下來的 sections 全部都是 meta-learning 的方法**

### Notions
- $\theta$: meta-learner 的參數
- $\phi_{T_s}$: meta-training task $T_s$ 的 task-specific 參數
- $\phi_{T_t}$: meta-testing task $T_t$ 的 task-specific 參數

由 $\theta$ 參數化的 meta-learner(optimizer) 提供 information 給 $\phi_{T_s}$ 參數化的 learner(optimizee)，然後 learner 就回傳 error signal 例如 gradient 給 meta-learner 以求進步。

於是，給定一個 meta-testing task $T_t$ 和 $D_{T_t}$，meta-learner 可以直接被使用，而 learner 可以從 $D_{T_t}$ 學習。

### 5.2.1 Refine by Gradient Descent
- **Model-Agnostic Meta-Learning (MAML)** 是具代表性的一個方法
    - Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. ICML 2017
    - meta-learn 一個 $\theta$ 當作一個很好的 initialization $\phi_{T_s}^0$
    - MAML 忽略了 task-specific information
    - 因此只適合非常相似的 task，當 task 差很多的時候很糟糕
- Gradient-Based Meta-Learning with Learned Layerwise Metric and Subspace. 2018, ICML
    - 學習從 $\theta$ 中選擇 subset 做為 $\phi_{T_t}^0$ 的 initialization (???原文很奇怪)
    - 即 meta-learn 一個 task-specific subspace and metric
    - 因此不同的 $T_t$ 有不同的 initialization $\theta$


### 5.2.2 Refine in Consideration of Uncertainty

用 few example 來 learn 必定導致很高的 uncertainty。學習好的 model 能否在新的 task 上有更高的 confidence? 如果有更多 samples，model 能變更好嗎? 衡量 uncertainty 的能力可以提供一些信號給 active learning 或者更進一步的 data collection。
- 論文：Probabilistic model-agnostic meta-learning. NeurIPS 2018

1. Uncertainty over the shared parameter $\theta$
    - 單一的 $\theta$ 可能不會對所有 tasks 都是好的 initialization，因此 modeling $\theta$ 的 posterior，可以對不同的 $T_t$
    - 論文：Probabilistic model-agnostic meta-learning. NeurIPS 2018
        - 該文提出要 model $\theta$ 的 prior distribution，
    - Bayesian model-agnostic meta-learning. NeurIPS 2018
        - 利用 Stein Variational Gradient Descent (SVGD) 來學習 $\theta$ 的 prior distribution

2. Uncertainty over the task-specific parameter $\phi_{T_t}$


3. Uncertainty over class $n$'s class-specific parameter $\phi_{T_t, n}$

### 5.2.3 Discussion

## 5.3 Learn Search Steps


## 5.4 Summary


# 6 FUTURE WORKS

## 6.1 Problem Setup

## 6.2 Tehiniques

## 6.3 Applications
Omniglot 跟 miniImageNet 已經有很高的 accuracy，新的 benchmark 是
- Meta-dataset: A dataset of datasets for learning to learn from few examples.
- A Simple Neural Attentive Meta-Learner. 2018, ICML
## 6.4 Theories

# 7 CONCLUSION


# Appendix Meta-Learning


---


[^oneshot06]: L. Fei-Fei, R. Fergus, and P. Perona. 2006. One-shot learning of object categories. IEEE Transactions on Pattern Analysis and Machine Intelligence 28, 4 (2006), 594–611. 
[^singleexp05]: M. Fink. 2005. Object classification from a single example utilizing class relevance metrics. In Advances in Neural Information Processing Systems. 449–456.
[^matchNet]: O. Vinyals, C. Blundell, T. Lillicrap, D. Wierstra, et al. 2016. Matching networks for one shot learning. In Advances in Neural Information Processing Systems. 3630–3638.
[^fewRetrieval]: E. Triantafillou, R. Zemel, and R. Urtasun. 2017. Few-shot learning through an information retrieval lens. In Advances in Neural Information Processing Systems. 2255–2265.
[^fewTrack]: L. Bertinetto, J. F. Henriques, J. Valmadre, P. Torr, and A. Vedaldi. 2016. Learning feed-forward one-shot learners. In Advances in Neural Information Processing Systems. 523–531.
[^surveyFewshot]: J. Shu, Z. Xu, and D Meng. 2018. Small sample learning in big data era. arXiv preprint arXiv:1808.04572 (2018).
[^memoNet]: A. Santoro, S. Bartunov, M. Botvinick, D. Wierstra, and T. Lillicrap. 2016. Meta-learning with memory-augmented neural networks. In International Conference on Machine Learning. 1842–1850.
[^neuralStatistician]: H. Edwards and A. Storkey. 2017. Towards a Neural Statistician. In International Conference on Learning Representations.


###### tags: `fewshot learning`



