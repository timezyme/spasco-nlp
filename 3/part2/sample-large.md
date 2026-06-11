# (How) Do Language Models Track State?

Belinda Z. Li <sup>1</sup> Zifan Carl Guo <sup>1</sup> Jacob Andreas <sup>1</sup>

## Abstract

Transformer language models (LMs) exhibit behaviors—from storytelling to code generation that appear to require tracking the unobserved state of an evolving world. How do they do so? We study state tracking in LMs trained or finetuned to compose permutations (i.e., to compute the order of a set of objects after a sequence of swaps). Despite the simple algebraic structure of this problem, many other tasks (e.g., simulation of finite automata and evaluation of boolean expressions) can be reduced to permutation composition, making it a natural model for state tracking in general. We show that LMs consistently learn one of two state tracking mechanisms for this task. The first closely resembles the "associative scan" construction used in recent theoretical work by [Liu et al.](#page-9-0) [\(2023\)](#page-9-0) and [Merrill et al.](#page-10-0) [\(2024\)](#page-10-0). The second uses an easy-to-compute feature (permutation parity) to partially prune the space of outputs, then refines this with an associative scan. The two mechanisms exhibit markedly different robustness properties, and we show how to steer LMs toward one or the other with intermediate training tasks that encourage or suppress the heuristics. Our results demonstrate that transformer LMs, whether pretrained or fine-tuned, can learn to implement efficient and interpretable state tracking mechanisms, and the emergence of these mechanisms can be predicted and controlled.[1](#page-0-0)

## 1. Introduction

Language models (LMs) are trained to model the surface form of text. A growing body of work suggests that these models nevertheless learn to represent the latent *state of the world*—e.g. situations described by language and results of program execution—to support prediction. However, the

<span id="page-0-0"></span><sup>1</sup>Code and data are available at [https://github.com/](https://github.com/belindal/state-tracking) [belindal/state-tracking](https://github.com/belindal/state-tracking)

mechanisms that LMs use to construct these representations are not understood. Do LMs simulate state evolution step-by-step across successive hidden layers or token representations [\(Yang et al.,](#page-10-1) [2024\)](#page-10-1)? Are states approximated via a complex collection of heuristics [\(jylin04 et al.,](#page-9-1) [2024\)](#page-9-1)? Is state tracking an illusion [\(Bender & Koller,](#page-9-2) [2020\)](#page-9-2)?

This paper studies the implementation and emergence of state tracking mechanisms in language models using permutation composition as a model system: given a fixed set of objects, we train or fine-tune LMs to predict the final position of each object after a sequence of rearrangements. Previous work has used versions of this task to evaluate LMs' empirical state tracking abilities [\(Li et al.,](#page-9-3) [2021;](#page-9-3) [Kim](#page-9-4) [& Schuster,](#page-9-4) [2023;](#page-9-4) [Li et al.,](#page-9-5) [2023\)](#page-9-5). Additionally, as shown by [Barrington](#page-9-6) [\(1986\)](#page-9-6), many complex, natural state tracking tasks—including simulation of finite automata and evaluation of Boolean expressions—can be reduced to permutation tracking with five or more objects. This makes it a natural model for studying state tracking in general.

Our analysis proceeds in several steps. Section [2](#page-2-0) provides technical preliminaries: [§2.1](#page-2-1) and [§2.2](#page-2-2) introduce state tracking problems and the permutation composition task we use to model them (Figure [1A](#page-1-0)), and [§2.3](#page-2-3) reviews the set of interpretability tools we use to analyze LM computations. Next, Section [3](#page-3-0) lays out a family of algorithms that past work has suggested LMs might, in principle, use to solve the state tracking task (Figure [1D](#page-1-0)), and describes the signatures—expected readouts from different interpretability methods—that we would expect to find if a given algorithm is implemented (Figure [1B](#page-1-0)-C).

Finally, Sections [4](#page-5-0) and [5](#page-7-0) present experimental findings. Across a range of sizes, architectures, and pretraining

schemes, we find that LMs consistently learn one of two state tracking mechanisms. The first mechanism, which we call the "associative algorithm" (AA), resembles the associative scan construction used by [Liu et al.](#page-9-0) [\(2023\)](#page-9-0) and [Merrill & Sabharwal](#page-10-2) [\(2024\)](#page-10-2) to establish theoretical lower bounds on the expressive capacity of Transformers. The second mechanism, which we call the "parity-associative algorithm" (PAA), first rules out a subset of final states using an easy-to-compute permutation parity heuristic, then uses an associative scan to obtain a final-state prediction. Notably, we fail to find evidence for either step-by-step

<sup>1</sup>MIT EECS and CSAIL. Correspondence to: Belinda Z. Li <bzl@mit.edu>.

![](_page_1_Figure_0.jpeg)

Figure 1. We use permutation word problems as a simple model of state tracking. Here actions permutations, and states are the products of those permutations; the current state can be tracked by taking the cumulative product from left to right (§2). We identify several possible algorithms that Transformers may use to solve permutation word problems, which we call sequential, parallel, associative, and parity-associative (§3). Above, we depict the "signatures" of each algorithm under two types of interpretability analysis: *prefix patching*, where all the activations are corrupted except the prefix up to a token at a particular layer, and *probing*, where we train a linear probe to map from last-token representations across the layers to either the final state or the final state parity (§2.3).

comp. theory. AUTOMATA: Are there any results on the state of the initial state of the minimal DFA for the union of two regular languages? (View)

simulation or for fully parallel composition mechanisms, even when these are theoretically implementable by LMs. We support our findings with evidence from representation interventions (Meng et al., 2022; Zhang & Nanda, 2024; §4.2), probes (Shi et al., 2016; §4.3), patterns in prediction errors (Zhong et al., 2024; (§4.4), attention maps (Clark et al., 2019; §4.5), and training dynamics (McCoy et al., 2019; Olsson et al., 2022; Hu et al., 2023; §5.1).

The scan operation for PAA appears difficult for LMs to im-

<span id="page-1-0"></span>plement robustly, and the choice of mechanism sometimes significantly impacts model performance on long sequences (§5.1). Whether a given LM learns AA or PAA is highly stochastic (§5.2). However, each is associated with a characteristic set of phase transitions in the training loss (Chen et al., 2024), and LMs can be steered toward one solution or the other by training on an intermediate task that encourages or discourages LMs from learning a parity heuristic (§5.3).

As pretrained LMs sometimes re-use circuits when fine-

tuned on related tasks [\(Prakash et al.,](#page-10-9) [2024;](#page-10-9) [Merullo et al.,](#page-10-10) [2024\)](#page-10-10), our results suggest a possible mechanism by which real-world LMs might perform state tracking when modeling language, code, and games. Looking beyond state tracking, these findings underscore both the complexity and variability of LM solutions to complex tasks, which may involve *both* heuristic features and structured solutions.

## <span id="page-2-0"></span>2. Background and Preliminaries

### <span id="page-2-1"></span>2.1. State Tracking

Inferring common ground in discourse [\(Li et al.,](#page-9-3) [2021\)](#page-9-3), navigating the environment [\(Vafa et al.,](#page-10-11) [2024\)](#page-10-11), reasoning about code [\(Merrill et al.,](#page-10-0) [2024\)](#page-10-0), and playing games [\(Li](#page-9-5) [et al.,](#page-9-5) [2023;](#page-9-5) [Karvonen,](#page-9-10) [2024\)](#page-9-10) all require being able to track the evolving state of a real or abstract world. There has been significant interest in understanding whether (and how) LMs can perform these tasks. In theoretical work, researchers have observed that many natural state-tracking problems (including the ones listed above) are associated with the complexity class NC<sup>1</sup> , and have shown that inputs of size n can be modeled by Transformers with O(log n) depth [\(Liu](#page-9-0) [et al.,](#page-9-0) [2023;](#page-9-0) [Merrill & Sabharwal,](#page-10-2) [2024\)](#page-10-2). Empirical work, meanwhile, has found that large LMs learn to solve state tracking problems [\(Kim & Schuster,](#page-9-4) [2023\)](#page-9-4) and encode state information in their representations [\(Li et al.,](#page-9-3) [2021;](#page-9-3) [Li et al.,](#page-9-5) [2023\)](#page-9-5). But a mechanistic understanding of *how* trained LMs infer these states has remained elusive.

## <span id="page-2-2"></span>2.2. Permutation Group Word Problems

Toward this understanding, the experiments in this paper focus on one specific state tracking problem, *permutation composition*. At a high level, this problem presents LMs with a set of objects and a sequence of reshuffling operations; LMs must then compute the final order of the objects after all reshufflings have been applied (Figure [1A](#page-1-0)). Though less familiar than discourse tracking or program evaluation, [Kim](#page-9-4) [& Schuster](#page-9-4) [\(2023\)](#page-9-4) used a version of this task to evaluate LM state tracking. More importantly, as shown by [Bar](#page-9-6)[rington](#page-9-6) [\(1986\)](#page-9-6) and recently highlighted by [Merrill et al.](#page-10-0) [\(2024\)](#page-10-0), permutation tracking (with five or more objects) is NC<sup>1</sup> -complete, meaning *any* other state tracking task in this family can be converted into a permutation tracking class. This, combined with its simple structure, makes it a natural model system for studying state tracking in general.

More formally, the finite symmetric group S<sup>n</sup> comprises the set of permutations of n objects equipped with a composition operation. For example, 42315 denotes the permutation of 5 objects (i.e. in S5) that moves the first object to the fourth position, the second object to the second position, etc. Importantly for our findings in this paper, every permutation can be expressed as a composition of two-element

swaps (in Figure [1A](#page-1-0), a0, but not a1, is an example of a swap). The parity of a permutation (even or odd) is the parity of the number of swaps needed to create it.

The composition of two permutations, standardly denoted a<sup>1</sup> ◦ a0, is the result of applying a<sup>1</sup> after a0. Inputs to sequence models in machine learning are typically written with earlier inputs before later inputs (i.e. left-to-right), so for consistency with this convention we will write a0a<sup>1</sup> to denote the application of a<sup>0</sup> *then* a1. Figure [1A](#page-1-0) shows the result of composing 42315 and 12534 in sequence.

Finally, the word problem on S<sup>n</sup> is the problem of computing the product of a sequence of permutations. This product itself corresponds to a single permutation (32514 in Figure [1\)](#page-1-0). But, following the intuition given at the beginning of the section, it may equivalently be interpreted as the final ordering of the objects being rearranged (DBAEC in Figure [1A](#page-1-0)). In accordance with this intuitive explanation (and by analogy to other state tracking problems), we will use a<sup>t</sup> to denote a single permutation ("action") in a sequence, and s<sup>t</sup> = a<sup>0</sup> · · · a<sup>t</sup> to denote the result of a sequence of permutations (a "state").

Given a sequence of permutations, we use ϵ(at) to denote the parity of the tth permutation, so:

<span id="page-2-4"></span>
$$\epsilon(s_t) = \epsilon(a_0 \cdots a_t) = \sum_i \epsilon(a_i) \mod 2$$
 (1)

(where ϵ is 0 for even permutations and 1 for odd ones).

All experiments in this paper train transformer language models to solve the word problem: they take as input a sequence of actions [a0, . . . , at], and output a sequence of state predictions [s0, . . . , st].

## <span id="page-2-3"></span>2.3. Interpretability Methods

Our experiments employ several interpretability techniques to understand how LMs solve permutation word problems, which we briefly describe below. Throughout this paper, we use ht,l to denote the internal LM representation at token position t after Transformer layer l, with T and L denoting the maximum input length and number of layers respectively.

Probing In probing experiments [\(Shi et al.,](#page-10-5) [2016\)](#page-10-5), we fix the target LM, then train a smaller "probe" model (e.g. a linear classifier) to map LM hidden representations h to quantities z hypothesized to be encoded by the LM (Figure [1C](#page-1-0)). Our experiments specifically evaluate whether (1) the state st, and (2) the final state parity is linearly encoded in intermediate-layer representations. For each layer l, we train (1) a state probe to predict p(s<sup>t</sup> | ht,l) ; and (2) a parity probe to predict p(ϵ(st) | ht,l). Given a trained LM, we collect representations on one set of input sequences to train the probe, then evaluate probe accuracy on a held-out set.

Activation Patching Probing experiments reveal what information is *present* in an LM's representations, but not that this information is *used* by the LM during prediction. Activation patching is a method for determining which representations play a *causal* role in prediction. Portions of the LM's internal representations are overwritten ("patched") with representations derived from alternative inputs; if predictions change, we may conclude that the overwritten representations was used for prediction (Meng et al., 2022; Zhang & Nanda, 2024; Heimersheim & Nanda, 2024).

Let  $p(y \mid x; h \leftarrow h')$  denote the probability that an LM assigns to the output y given an input x, but with the representation h replaced by some other representation h'. In a typical experiment, we first construct a "clean" input x that we wish to analyze, and a "corrupted" input x' that alters or removes information from x (e.g. by adding noise or changing its semantics). Next, we compute the most probable outputs from clean and corrupted inputs:

$$\widehat{y} = \arg \max_{y} p(y \mid x)$$
$$\widehat{y'} = \arg \max_{y} p(y \mid x')$$

We then re-run the LM on the corrupted input x', but substitute a hidden representation from the clean input x, and measure how much prediction shifts toward the clean output  $\hat{y}$  using the normalized logit difference (Wang et al., 2022):

$$NLD = \frac{LD(x'; h_{t,l} \leftarrow h_{t,l}^{clean}) - LD(x')}{LD(x) - LD(x')}$$
(2)

where

 $LD(\cdot) = \log p(\hat{y} \mid \cdot) - \log p(\hat{y'} \mid \cdot)$ 

and the representation  $h_{t,l}^{\text{clean}}$  is taken from the clean run of the model. A value of NLD close to 1 indicates that we have *restored* a part of the circuit that computes  $\hat{y}$ .

In this paper, we evaluate which representations are involved in prediction by presenting models with a clean sequence  $[a_0, a_1, \ldots, a_t]$  associated with a final state  $s_t$ . We then produce a corrupted sequence  $[a'_0, a_1, \ldots, a_t]$ , associated with a final state  $s'_t$ , and identify the hidden states that, when patched in, cause the model to output  $s_t$  rather than  $s'_t$  with high probability. Our main experiments specifically

<span id="page-3-0"></span>
