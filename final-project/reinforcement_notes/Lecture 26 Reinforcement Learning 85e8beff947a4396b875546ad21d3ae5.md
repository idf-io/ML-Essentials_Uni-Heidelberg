# Lecture 26 Reinforcement Learning

- learning method for agents that can act e.g.,
- two crucial differences to supervised learning
    1. SL is passive only makes predictions
    RL is active use predictions to act ⇒ changes outcome in real world
    2. SL has strong supervision = GT outcome for every instance
    RL has weak supervision = feedback is rare

Example chess: SL = student & a coach: coach tells for every move if and why it was good or bad

RL = only feedback is win/loss/draw at the end (1.5 bits per game) ⇒ for loss you don’t know which move was the big mistake

Theory and methods now make the best of scarce feedback

reinforce behavior leading to positive outcomes

1. agent’s action influence world:
2. world’s feed on action quality is week (chess: win/loss/draw, $\log_23$ = 1.5 bits gene) 

feedback at time t is given by reward $R_t$ (positive: good, gain / negative: bad, penalty / zero: neutral no feedback at time t; very often)

Can be improved by reward shaping: introduce auxilliary rewards to get more feedbacks

[ chess  

apply with care, because it may change the optimal solution ⇒ Hieoretical advice later

## Formal Definitions

- At time t, environment/“world” state $s_t$ action $a_t$ rewards $R_t$ observations $o_t$
- use discrete: $s_0\rightarrow s_1\rightarrow \dots \rightarrow s_t\rightarrow\dots s_T$
    - $T\rightarrow \infty$ “never ending”
    - $T$ finite: “episode”
- Markov property: next state $s_{t+1}$ only depends on last state $s_t$, rather than past $s_{t-1},s_{t-2}$ etc. mathematically, $s_{t-1}\perp s_{t+1}|s_t$ (Markov independence relation)
[If violated, can often be rescued by adding some memory to $s_t$] e.g., chess remember if casting is allowed
- if fulfilled Markov chain $s_0\rightarrow s_1\rightarrow \dots \rightarrow s_T$
    
    ⇒ joint probability of the world
    

Often, $s_t$ cannot be observed fully (card games opponent’s cards) ⇒ observation probabilities $p(O_t|s_t)$

- reward for action $a_t$ is paid after transition $s_t\rightarrow s_{t+1}\rightarrow R_{t+1}$

## Complete models

Markov Decision Process (MDP) and Partially observe MDP (POMDP)

![Untitled](Lecture%2026%20Reinforcement%20Learning%2085e8beff947a4396b875546ad21d3ae5/Untitled.png)

Full transition probability $p(s_{t+1},R_{t+1}|s_t,a_t)$ fully describes behavior of the “world”

- “Return” total reward after time $t$ up to the and $G_t = \sum_{t'=t+1}^{T}R_{t'}$
”discounted return” downweighted the (discounted) future necessary for endless games $G_t=\sum_{t'=t+1}^T\gamma^{t'-t-1} R_t$
- “policy” rule how the ugent chooses actions $a_t$
    - determinisitic (function): $a_t=\prod (s_t)$ ( or $\prod (o_t)$)
    - randomized (conditional probability) $a_t \sim p(A_t|s_t)$ (or $p(A_t|o_t)$)
- goal approximate optimal policy $\hat\prod (s)\approx \prod^*(s)$ ← maximize expected return
- value function given some policy what’s expected return in state $s_t$? (p: world probability; $\pi$: policy)

$$
V_{\pi}(s)=\mathbb E_{p,\pi}[G_t|s_t=s]=\sum_a\pi(a|s)p(s,a)
$$

- Optimal policy: $\pi^*_s=\arg\max_{\pi}V_{\pi} (s)$ optimal value function $V^*(s)=V_{\pi^*}(s)$
- when state and action space are discrete ⇒ write expectation as a sum
    
    $V_{\pi}(s)=\sum_a\pi(A=a|s)\sum_{s',R}p(s',R|s,a)\cdot G_+$ , where:
    
    - $a$ all legal actions in state S
    - $S'$: possible subsequencial states
    - $R$: possible rewards
    - $G_+=R+\mathbb E_{p,\pi} [\gamma G_{t+1}|s_{t+1}=s_t']=R+\gamma V_\pi (s')$
- Bellman equations for value function:
    
    $$
    V_{\pi}(s)=\sum_a\pi(A=a|s)\sum_{s',R}p(s',R|s,a)[R+\gamma V_\pi(s')]
    $$
    
    - Valid value function must be self-consistent ⇒ if $p(s',R|s,a)$ is known ⇒ linear system of equations
    - solve by any linear solver e.g., iteratively (Gauss-Seidel?) ⇒ value iteration
    
    > Gauss-Seidel is an iterative method used to solve a system of linear equations
    > 
1. initial guess for $V_\pi^{(0)}$ (e.g., $V_\pi^{(0)}(s)=s$)
2. for $\gamma =1,2,....$ (to convergence)
    
    for all s, $V_{\pi}^{(t)}(s)=\max_{a}\sum_{s',R}p(s',R|s,a)[R+\gamma V^{(t-1)}_\pi(s')]$
    
    converges to optimal policy, $\pi^*(s)=\arg\max_{a}\sum_{s',R}p(s',R|s,a)[R+\gamma V^*_\pi(s')]$
    
- catch $p(s_{t+1}, R_{t+1}|s_t, a_t)$ is usually known
    - model-based RL learn $U^*(\pi^*)$ jointly with $p(s',R|s,a)$ ⇒ hard
    - model-free RL： instead of learning p and calculating expectations  ⇒ learn expectations directly “Q-learning”

Q-function or **“action-value” function** return expected value of states, if we take action $a$, but then follow policy afterwards

$$
V_{\pi}(s)=\sum_a\pi(A=a|s)\sum_{s',R}p(s',R|s,a)[R+\gamma V_\pi(s')]=\sum_a\pi(A=a|s)Q_\pi(a,s)
$$

optimal value $V_\pi(s)=\max_a Q(s,a)$

Bellmann equation for Q-function (optimal) can be learned from data alone without knowing $p(s',R|s,a)$

$$
Q^*(s,a)=\sum_{s',a}p(s',a|s,a)[R+\gamma \max_{a'}Q^*(s',a')]
$$

### Basic Q-learning algorithm

repeat very often

1. play game according to current plicy
2. use rewards to update Q-function (don’t learn the probability p)
3. update policy from new Q-function

Exploration-exploitation trade-off

two conflicting task during learning

1. improve accuracy of “$Q(s,a)$” for the actions we already tried ⇒ better policy $\pi(s)=\arg\max_a Q(s,a)$ “exploitation”
2. discover actions that load to new successful strategies “exploration”

**Illustration (trade-off)**

![Untitled](Lecture%2026%20Reinforcement%20Learning%2085e8beff947a4396b875546ad21d3ae5/Untitled%201.png)

adjust trade-off $\epsilon-$ greedy policy $\pi(A|s)$ is current guess

select action $a$ 

- $\sim \pi(A|s)$ with probability $1-\epsilon$ (where $\epsilon$ is a hyperparameter)
- $\sim uniform(A)$ with probability $\epsilon$

Theorem: after infinite time ,every move was tried infinitely often

Softmax policy, $Q(s,a)$ current guess

$$
a \sim softmax(Q) \Leftrightarrow P(A=a|s)=\frac{\exp(Q(s,a)/\rho}{\sum_{a'}\exp Q(s,a')/\rho}
$$

- $\rho=0$ ⇒ determine policy $a=\arg\max_{a'} Q(s,a')$
- $\rho=\infty$ ⇒ purely random policy $a\sim uniform(A)$

## Q-learning-main RL method

- return after step t: $G(t)=\sum_{t'=t+1}^{T}\gamma^{t'-t-1}R_{t'},\quad \gamma\in(0,1]$
- policy: $\left.\begin{array}{l}a=\pi(S) \\a \sim \pi(A \mid S)\end{array}\right\}$ rule to select actions in state $s$
- value function
    
    expected return of states when using policy $\pi$
    
    $$
    V_{\pi}(s)=\mathbb E_{\rho,\pi}[G_t]
    $$
    
- action-value function: expected return in state $s$ when starting with action $a$ and then playing with policy $\pi$:
    
    $$
    Q_\pi(s,a)=\mathbb E_{s',R\sim p(s',R|s,a)}[R+\gamma V_{\pi}(s')]
    $$
    
- for optimal policy,
    
    $$
    \pi^+(s)=\arg\max_aQ^*(s,a)
    $$
    
    where $Q^*$ is the optimal Q-function from Bellman equations:
    
- for suboptimal policy, we can still define new policy:
    
    $$
    \pi'(s)=\arg\max_aQ_\pi(s,a)
    $$
    

where $Q_{\pi}(s,a)$ learned from old policy $\pi(s)$

Theorem: $\pi'(s)$ is never worse than $\pi(s)$ ⇒ possibility of “off-policy” learning = use training data from a bad policy later in training for better policy ⇒ iterate for long time $\pi'\rightarrow \pi^*$

- **Goal of Q-learning**: train a good policy $\pi$ without first learning the behavior of the world $p(s',R|s,R)$
    
    “model-free RL”
    
- trick: approximate expectation by a single instance
”temporal difference learning”
    
    $$
    \begin{aligned}
    Q^*(s,a)&=Q_\pi(s,a)+[Q^*(s,a)-Q_\pi(s,a)] \\
    &=Q_\pi(s,a)+[\mathbb E_p[R+\gamma V^*(s')]-Q_\pi(s,a)] \\
    &\approx  Q_\pi(s,a)+[R+\gamma V_\pi(s')-Q_\pi(s,a)]
    \end{aligned}
    $$
    

Idea of algorithm

1. collect training data with current policy $\pi$
2. update Q-function
    
    $$
    Q'(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha [R_{t+1}+\gamma V_\pi (s_{t+1})-Q_\pi(s_t,a_t)]
    $$
    
3. update policy
4. repeat

Two major variant

- Q-learning (narrow sense) $V_\pi(s)=\max_a Q_\pi(s,a)$
- SARSA algorithm $V_\pi(s_t)=Q_\pi(s_t,a_t)$

Q-learning on SARSA algorithm

1. initial guess $Q^{(0)}(s,a)$ (good guesses speed up convergence)
2. outer loop $\gamma=1,\dots,E$ (play many guesses)
    1. innter loop $t=1,T_\tau$ (steps of game $\tau$)
        1. select next action according to 
            - $\epsilon$-greedy policy
                
                $$
                a_{\tau,t}=\left \{\begin{aligned} 
                &\arg\max_aQ^{(\tau-1)}(s_t,a) & \text{with prob.} (1-\epsilon) \\
                &uniform(A) & \text{with prob.} \epsilon
                \end{aligned}\right.
                $$
                
            - softmax policy
            
            $$
            a_{\tau,t}\sim softmax(Q^{(\tau-1)}(s_t,a))
            $$
            
    2. for $t=1,\dots, T_\tau$
        1. update Q-function
            
            $Q^{(\tau)}(s_{\tau,t},a_{\tau,t})\leftarrow Q^{(\tau-1)}(s_{\tau,t},a_{\tau,t}) +\alpha[R_{\tau,t+1}+\gamma V^{(\tau-1)}(s_{\tau,t+1})-Q^{(\tau-1)}(s_{\tau,t},a_{\tau,t})]$
            where $R_{\tau,t+1}+\gamma V^{(\tau-1)}(s_{\tau,t+1})-Q^{(\tau-1)}(s_{\tau,t},a_{\tau,t})$ is **temporal difference (TD)** error
            
3. final policy
$\hat{\pi}(s)=\arg\max_aQ^{(E)}(s,a)$ or $\hat \pi(A,S)=softmax(Q^{(E)}(s,a))$ for low

- n-step Q-learning
more accurate updates by using multiple steps of the game $Q^{(\tau)}(s_{\tau,t},a_{\tau,t})\leftarrow Q^{(\tau-1)}(s_{\tau,t},a_{\tau,t}) +\alpha[\sum_{t'=t+1}^{t+n} \gamma ^{t'-1-1}R_{\tau,t+1}+\gamma^n V^{(\tau-1)}(s_{\tau,t+1})-Q^{(\tau-1)}(s_{\tau,t},a_{\tau,t})]$
- experience buffer
collection of training data from many games (possibly proved to keep size tractable)
⇒ update step b. use a random batch from buffer
    - tricks:
        - don’t put near steps from same game into same batch to avoid biased updates due to correlation
        - sample instances with higher probability where current guess of Q is very bad: $(G_{\tau,t}-Q^{(\tau-1)}(s_{\tau,t},a_{\tau,t}))^2\gg 0$
    - problem: if done naively, $Q(s,a)\in \mathbb R^{|S|\times|A|}$ is a very big matrix where S and A are number of states and actions respectively.
        - Q cannot be stored in memory
        - We cannot collect enough training data to estimate all elements
        
        ⇒ need regression method the generalize the TS
        

1. reduce size of state space by clustering similar states 
similar := optimal actions are similar 
example: Bomberman game rotationally & mirror symmetric
2. reduce Q-learning to a regression problem $\hat y =f(x)$:
    - $X$: the state $S$ itself, $\phi(s)$ hand-crafted or learned feature space of $S$
    - $y$: vector of length $|A|$ with expected return for every action

$y=f(x,a)$: y - scaler (return of action a)

common ways to define ground-truth response for TS

- Monte-Carlo value estimation: use actually observed rewards (accurate but requires lots of training data)
$y_{\tau,t}=\sum_{t'=t+1}^{T'_\tau}\gamma^{t'-t-1}R_{t'}$
- temporal difference (TD) Q-learning: $y_{\tau,t}=R_{T,t+1}+\gamma \max_a Q^{(\tau-1)}(s_{\tau,t+1},a)$
- n-step TD Q-learning: $y_{\tau,t}=\sum_{t'=t+1}^{t+n} \gamma ^{t'-1-1}R_{\tau,t+1}+\gamma^n V^{(\tau-1)}(s_{\tau,t+1})-Q^{(\tau-1)}(s_{\tau,t},a_{\tau,t})$
- SARSA
$y_{\tau,t}=R_{\tau,t+1}+\gamma Q^{(\tau-1)}(s_{\tau,t+1},a_{\tau,t+1})$

linear regression (only works with very good features $X=\phi(s)$) $\hat Q(s,a)=\phi(s)\beta_a$

train by batch gradient descent

1. sample batch from experience buffer, where $a_{\tau,t}=a$
2. update: $\beta_a\leftarrow \beta_a+\frac{\alpha}{N_{batch}}\sum_{\tau,t\in batch}\phi(s_\tau,t)(y_{\tau,t}-\phi(s_{\tau,t})\beta_a)$

non-linear regression with regression forest

- can train one forest per action, or a single returning a vector
- still needs good features, but nearly as good as linear regression

Deep Q-learning: use neural network

- typically, set $X=s$ to exploit
- example DeepMind’s solution to ATARI games
X= image of video game output ⇒ use CNN
- disadvantage: finding good network architecture and train to convergence may take longer than the deadline

## Reward shaping

- add auxilliary rewards during training when official rewards are very sparse
- idea:
    - define intermediate goals (e.g. “taking queen is usually good” unless a trap)
    - assign rewards for achieving intermediate goals
- must be used with care because it may change optimal policy ⇒ agent behaves poorly without auxilliary rewards in official game
- theory:
    - optimal policy does NOT change if auxilliary rewards are derived from a potential function $\psi(s)$
    - $\psi(s)$ does NOT depend on action take in state S 
    ⇒ auxilliary reward $F(s_{\tau,t},s_{\tau,t+1})=\gamma \psi(s_{\tau,t+1})-\psi(s_{\tau,t})$
    ⇒ new return $R'_{\tau,t+1}=R_{\tau,t+1}+F(s_{\tau,t}, s_{\tau,t+1})$
- optimal theoratical choice $\psi(s)=V^*(s)$, but dangerous during exploration

![Untitled](Lecture%2026%20Reinforcement%20Learning%2085e8beff947a4396b875546ad21d3ae5/Untitled%202.png)

# 课后补充

RL代表着**行为主义学派** behaviorism 与 DL 所代表的**连接主义学派 connectionism**，相互促进，共同繁荣

Reinforcement learning与多个学科交叉，mathematics, psychology, engineering, economics, neuroscience, computer science

通常一个比较困难的RL问题主要会在几个部分，深度学习能够帮助解决部分问题。

- State
- Action
- Reward

![Untitled](Lecture%2026%20Reinforcement%20Learning%2085e8beff947a4396b875546ad21d3ae5/Untitled%203.png)

![Untitled](Lecture%2026%20Reinforcement%20Learning%2085e8beff947a4396b875546ad21d3ae5/Untitled%204.png)

![Untitled](Lecture%2026%20Reinforcement%20Learning%2085e8beff947a4396b875546ad21d3ae5/Untitled%205.png)

## Outline

- Reinforcement Learning
    - Markov Decision Process (MDP)
- Value-based RL
    - Q-Learning (DQN)
- Policy-based RL
    - Policy Gradient (REINFORCE)
- Games
    - AlphaGo, AlphaGo Zero, AlphaZero, MuZero
- Real World

强化学习的目标函数本质上可以视作是一个动态规划问题，具有最优子结构的优化问题

随机逼近定理（强化学习的理论基础之一）：从一个不太好的Q可以逼近到比较不错的Q

采用Monte Carlo方法/Bellman 方程对Q来进行估计