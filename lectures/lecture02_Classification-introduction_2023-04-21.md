# Lecture 02: Classification introduction

Topics covered:
- [x] confusion matrix
- [x] error rate
- [x] false positive/negative rate,
- [x] precision/recall
- [x] threshold classifier



## Classification

Y is discrete
Without the loss of generatily: 

- Case: `C > 2`

  $Y = K \in 1, 2, ..., C$
    - C: number of classes/labels


- Special case: `C = 2`

  $Y \in \\{0,1\\}$ or $Y = \\{-1, +1\\}$
  - +1: Positive outcome
  - -1: Negative outcome


### Confusion matrix

|  $\hat{Y}$ \\ $Y^*$ | +1       | -1 |          |
| ------------------- | -------- | -- | -------- |
| +1                  | TP       | FP | FDR, precision |
| -1                  | FN       | TN |          |
|                     | TPR, FNR, recall | FPR| Accuracy |

- TP: True positive
- TN: True negative
- FP: False positive (Type I error)
- FN: False negative (Type II error)

### Performance measurement

- TPR: True positive rate $\frac{TP}{TP + FN}$
- FPR*: False positive rate $\frac{FP}{FP + TN} \approx p(\hat{Y} = 1 | Y^* = -1)$
- FNR*: False negative rate $\frac{FN}{FN + TP} \approx p(\hat{Y} = -1 | Y^* = 1)$
- FDR*: False discovery rate $\frac{FP}{FP + TP}$
- Accuracy $\frac{TP}{TP + FP + FP + FN}$

- False positive fraction $\frac{FP}{N} \in [0,1]$
- False negative fraction $\frac{FN}{N \in [0,1]}$

*: Ideally 0

### Classification types

- **Deterministic / Hard response**

  - There is a single discrete decision for each prediction
  
    $\hat{Y}_i = f(x_i)$  with $\hat{Y}_i = Y^*_i$
  
- **Probabilistic / Soft respose**

  - Returns a probability for every label = posterior distribution $p(Y=K|X)
  - More general than a hard classification: can recover decision function by returning the most probable label
  
    $p(Y=K|X) \text{ } \Rightarrow \text{ } f(x) = argmax_K$ $p(Y=K|X)$
    
  - Quality measured by "calibration"
    - If $p(Y=K|X)=v$, then the label should be correct v% of the cases
    - If actual accuracy higher v% $\rightarrow$ underconfident
    - If actual accuracy lower v% $\rightarrow$ overconfident
      - Dangerous, often in NN
      
> THM: Calculate confusion matrix or calibration from the test set


## Bayes rule of conditional probabilities

$$ p(Y=K|X) = \frac{p(X|Y=K)p(Y=K)}{p(X)} $$

- Prior probability: p(Y=K)
- Posterior probability: p(Y=K|X)
- Likelihood: p(X|Y=K)
- Evidence/Marginal: p(X)

### Origin

Joint distribution of features and labels: $p(X,Y)$ can be decomposed by chain rule of probability in two ways:
$p(X,Y) = p(Y|X)p(X) \Rightarrow$ First measure features, then response
or
$p(X,Y) = p(X|Y)p(Y) \Rightarrow$ First determine the response and then compatible features
We can derive:
$$p(X) = \Sigma_{K=1}^C p(Y=K)p(X|Y=K)$$

### Why is this important?

- It's a fundamental equation for probabilistic ML because it allows for clean uncertainty handling.
- Defines two fundamental model types: discriminative & general
- Puts model accuracy into perspective. What is good or bad? Confusion matrix

## Fundamental model types

- Discriminative

  - Learn p(Y=K|X)
  - Answer question directly: what class does the data belong to?
  - (+) Relatively easy - take direct route, no detour
  - (-) Often hard to interpret how model makes decisions. "Black box" behaviour of neural networks

- Generative

  - Learn $p(Y=K) \text{ and } p(X|Y=K)$
  - First learn how "the world works": understand mechanism, then use this to answer the question
    - E.g. how observations ("phenotypes") arise from hidden properties ("genotypes")
  - (-) More difficult: need powerful models and a lot of data
  - (+) More interpretable
  - History
    - Traditional science seeks generative models: they can create synthetic data that are often indistinguisable from real data
    - $\approx$ 1990: ML researchers realised that their models were too weak
    
      $\Rightarrow$ Field focused on discriminative models
      
    - $\approx$ 2012: Neural networks solved many hard discriminative tasks
    
      $\Rightarrow$  Field is again interested in generative models (ChatGPT, Midjourney - Subfield "explainable/interpretable ML"
      
      
## How does Bayes help to reevaluate accuracy?

**How good can a classifier be?**

  - Def.: Bayes classifier uses Bayes rule (left side or right side) with true prob. $p^*$

> THM: No learned classifier using $\hat{p}$ can be better than Bayes with $p^*$

  - Example:

    $p^*(Y=1|X)=0.6$ then error rate of less than 1 - 0.6 = 40% is **impossible**

**How bad can a classifier be?**

- Case 1: all classes are equally probable $p(Y=K)=\frac{1}{C}$ for all K=1,...,C

  Features are uninformative of the response:  
  $p(X|Y=K) = p(X|Y=K')$ for all K, K' pairs $\in {1, ..., C}$
  
  Example of two classes $p(Y=4)=0.5$  
  Worst classifier: pure guessing $\Rightarrow$ correct 50% of the time

- Case 2: unbalanced classes

  E.g.  
  $p(Y=1)=0.01 \Rightarrow$ Has disease  
  $p(Y=-1)=0.99 \Rightarrow$ Does not have disease  
  Worst classifier: always returns the majority label $\Rightarrow$ 99% correct
  
  Example:  
  Breast cancer screening test, e.g. mammography  
  $p(Y=1)=0.01 \Rightarrow$ Has disease  
  $p(Y=-1)=0.99 \Rightarrow$ Does not have disease  
  Worst case:  
  $p(X=\text{test positive}|Y=1) = 0.99$  
  $p(X=\text{test positive}|Y=-1)=0.01$  
  
  Question: if a test is positive, should you panic?  
  Using Baye's rule:  
  $$p(Y=1|X=\text{test positive} = \frac{p(X=1|Y=1)p(Y=1)}{p(X=1|Y=1)p(Y=1) + p(X=1|Y=-1)p(Y=-1}$$
  $$= \frac{0.99 \cdot 0.01}{0.99 \cdot 0.01 + 0.01 \cdot 0.99} = 0.5$$

  

## Threshold classifiers

Single feature: $X_i \in \mathbb{R}$ used to classify using a defined threshold (T)

$$
\hat{Y} = sign(X - T) =
\begin{cases}
  1 & \text{if } X > T \\
  -1 & \text{if } X < T
\end{cases}
$$

Usually a single feature is not enough to predict respose.

### Three classical multiple feature classifiers

1. Design a formula to combine features into a single score

    - ! Hard and expensive for most problems
    - E.g. BMI (body-mass-index)
    
2. **Linear classification**

    - Compute score as a linear combination and learn coefficients
    
      $S_i = \sum_{j=1}{D} \beta_j X_{ij} \text{ } \Rightarrow \text{ }$
      
      $\hat{Y}_i = sign(S_i - T)$


3. **Neares-neighbour (NN) classification**:  

    - Classify test instance $X_test$ according to most similar training instance
    
      $\hat{i} = argmin_i  \text{ } dist(X_i, X_{test})$
      
    - Problems:
      - Computationally expensive when N is large. They store the entire TS and scan for query $X_test$.
      Faster search exists byt scales badly with D
      - Very hard to define $dist(X_{test}, X_i)$ to reflect true semantic similarity
      
      $\Rightarrow$ machine learning task "metric learning" figure out dist from training data
