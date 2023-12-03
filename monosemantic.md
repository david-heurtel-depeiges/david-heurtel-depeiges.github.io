---
title: "Towards Monosemanticity : Decomposing Vision Models with Dictionary Learning"
summary: "About this page."
date: 2023-12-03
layout: default
---

### WORK IN PROGRESS

#### Quick Disclaimer
This is a research project that is still in progress. I am still working to improve the quality of the code and the clarity of the results. I do not have Anthropic's manpower or compute ressources. I will obviously leave some questions unanswered and this should be taken more as a proof of concept than a definitive answer to the question of monosemanticity in vision models. Most notably, I have not been able to examine fully the result of each model in each hyper-parameter sweep. And the metric we suggest is not perfect.

### Abstract

In artificial neural networks, the most basic computing unit, the neuron, can sometimes exhibit a phenomenon called superposition. This is the ability of a neuron to respond to very different features/inputs. If this is thought to increase the representational power of the network by learning over-complete representations and cramming more information in a single neuron, it can also hinders effort to probe said neural networks and hinder interpretability. In this work, we reproduce Anthropic's work on monosemanticity in language models and apply it to vision models. We show that their method is promising beyond its original scope. We also suggest a metric to evaluate the monosemanticity of a learned dictionary feature.

## Introduction

In 2017 in their Distill article [Feature Visualization](https://distill.pub/2017/feature-visualization/), Olah et al. uncovered neurons in InceptionV1 that were responding to wats, foxed and cars. When probing further and visualizing the features learned by said neuron, one could clearly recognize a patchwork of cats head and cars hoods and windshields. This is a clear example of superposition. Authors wrote:
>"Examples like these suggest that neurons are not necessarily the right semantic unit for understanding neural nets."

In language models trained by Anthropic, they similarly found neurons responding to a mixture of code and natural language. Probing superposition further in their series [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/), they suggest that an independent, monosemantic feature is actually a linear combination of neurons. They contend this increases neural network representation power, increases neural network ability to "represent more independent "features" of the data than it has neurons". Experiments on small toy models suggest this phenomenon arises naturally during training. 

On the flip side, Anthropic seems to have shown superposition is actually one of the main causes of neural network vulnerability to adversarial attacks. Indeed, if semanticly independent features are actually close linear combinations of neurons, a small perturbation in the input can cause a set of activations to "jump" from one mono-semantic feature to another. This is a very interesting result, but it is not the focus of this work. 

Superposition is also a problem for interpretability. If a neuron is responding to a mixture of features, it is harder to interpret what it is actually doing. This will be the focus of our work on vision models. We will try to implement one of Anthropic's strategy to find monosemantic features. They developed this method for a 1 layer transformer model. We will apply it to InceptionV1. Said method is a weak dictionary learning technique that relies on a sparse 1-layer deep auto-encoder. By having a very large bottleneck, we allow the auto-encoder to learn an overcomplete basis of neuron activations that is monosemantic. Sparsity, enforced via L1-regularization, is key in forcing the auto-encoder learn a monosemantic or "few-semantic" representation of the set of neuron activations. Indeed, L1-regularization forces the auto-encoder to reconstruct the input using only a few features/a few vectors in the overcomplete basis. Naturally, the features and vectors in question will be tailored to a specific set of neuron activations. 

In this work, we show that this method is promising beyond its original scope. We also suggest a metric based on word semanticity (of the ImageNet classes) to evaluate visual monosemanticity. As discussed, this metric is not perfect and makes more sense for the last layers of a vision classifier. For earlier layers, automatic feature evaluation might be done using CLIP embeddings. We leave this for future work and remain open to collaboration (see the TODO section in the Appendix).

## 1. Background and Methods

### 1.1 Background

Anthropic blog post presents a very good summary of previous work on dictionary learning, weak dictionary learning, sparse auto-encoders and how they were used in the past for interpretability reason. The reader should refer to their blog post for a complete overview (and also because their work is much more complete than ours, although done for LLMs). We will not repeat it here.

### 1.2 Architecture and Methods

We will work on InceptionV1 activations. More precisely, we sill work on layers 4a (middle layer, one of the first exhibiting superposition) and layer 5b (last layer before classification head). Here is a reminder of the architecture of InceptionV1:

<figure>
  <img style="float:none;" width="800" height="170" src="/docs/Inceptionv1L.png" alt="InceptionV1 Architecture">
  <figcaption>Figure 1: InceptionV1 Architecture - Layers 4a and 5b. Original figure by <a href="https://distill.pub/2017/feature-visualization/" target="_blank">Google Brain</a>.</figcaption>
</figure>

When focusing on a specific layer, we will compute activations after concatenation of all the branches. For a batch of shape $$(B, C_{in}, H_{in}, W_{in})$$, producing a batch of shape $$(B, C_{layer}, H_{layer}, W_{layer})$$, we then sum averages activations over the spatial dimensions to get a batch of shape $$(B, C_{layer})$$. These will be the activations we will work with. Obviously we remove spatial information by doing so and learned features without spatial averaging might be more interesting. Notably, one should perhaps examine whether there exist some form of spatial invariance in the learned features or if there is maybe some form of spatial superposition/polysemanticity. A fixed set of activations may not correspond to the same feature in the top or bottom of the image. But we are getting off topic.

#### 1.2.1 Auto-encoder

We then train an auto-encoder on the $$(B, C_{layer})$$ activations. Let's have a quick reminder of why this might be a good idea. We present here Anthropic's justification. We want to decompose the activation vector $$\mathbf{x}$$ of size $$C_{layer}$$ into a linear combination of $$K$$ features $$\mathbf{d}_k$$ of size $$C_{layer}$$:


$$
\begin{equation*}
\mathbf{x} \approx b + \sum_{k=1}^K f_k(\mathbf{x})\mathbf{d}_k
\end{equation*}
$$


where $$b$$ is a bias term and $$f_k(\mathbf{x})$$ is a scalar function corresponding to the activation of the $$k^{th}$$ feature. $$\mathbf{d}_k$$ is the $$k^{th}$$ feature direction. The $$\mathbf{d}_k$$ form the overcomplete basis we mentionned above. We impose that they have unit norm. We know need to determine or at least approximate the $$f_k(\mathbf{x})$$ functions. We do so by training the auto-encoder. The output of the encoder is:


$$
\begin{equation*}
f_k(\mathbf{x}) = \mathrm{ReLU}(\mathbf{W}_e(\mathbf{x} - b_d) + b_e)_k
\end{equation*}
$$


where $$\mathbf{W}_e$$ is the encoder weight matrix, $$b_d$$ is the decoder bias and $$b_e$$ is the encoder bias. One might notice that we first remove the decoder bias. $$b_d$$ will correspond to the $$b$$ term in the first equation of this paragraph.
The output of the decoder is then:


$$
\begin{equation*}
\mathbf{\hat{x}} = \mathbf{W}_d(f_1(\mathbf{x}), ..., f_K(\mathbf{x})) + b_d
\end{equation*}
$$


where $$\mathbf{W}_d$$ is the decoder weight matrix. We force column of $$\mathbf{W}_d$$ to have unit norm. This is equivalent to forcing the $$\mathbf{d}_k$$ to have unit norm. 

This auto-encoder is then trained with the following loss:


$$
\begin{equation*}
\mathcal{L} = \mathbb{E}(\vert\vert\mathbf{x} - \mathbf{\hat{x}}\vert\vert_2^2 + \lambda \vert\vert\mathbf{f}(\mathbf{x})\vert\vert_1)
\end{equation*}
$$


where $$\mathbf{f}(\mathbf{x}) = (f_1(\mathbf{x}), ..., f_K(\mathbf{x}))$$ and $$\lambda$$ is a sparsity parameter.

#### 1.2.2 Training

Training of the auto-encoder is challenging. Most notably, some feature die during training. This was known to happen in Anthropic's work. Periodically, they would "revive" dead features by reinitializing their weights. We do the same. During training, we compute the frequency of activation of each feature over the whole dataset. If a feature activation frequency falls bellow a certain threshold, we reinitialize its weights. Unfortunately, this scheme was very agressive and seems to harm significantly the L2 reconstruction loss. For a discussion about this phenomenon, see the Appendix.

### 1.3 Analyzing the results

#### 1.3.1 Visual examination of features

The first analysis of the result is manual and visual. We look at a learned feature and the images that activate it the most. We then try to interpret the feature. Conversely, we also take a class or a group of classes (like "dogs" in general) and then look at the features that are the most activated by instances of the class. We then verify that images that activate these features in general have some common semantic property with class instances.

#### 1.3.2 Suggestion of a new metric

Visually examining features is very time consuming and subjective. Ideally, one would have an automated way of evaluating the monosemanticity of a feature. In Anthropic's work, they actually used their own LLM, Claude, to evaluate how monosemantic learned language features were (and Claude was in agreement with human assessment). We do not have an equivalent model.

ImageNet classes, although not a perfect description of our world are actually synsets in the WordNet database/graph. This means that we have a graph of relations (hypernyms, hyponyms...) between ImageNet classes and several other synsets in the WordNet database. Using that, we can build a minimum spanning tree, corresponding to ImageNet hierarchy, with each leaf node corresponding to an ImageNet class and each child of a node being a more precise instance of the father node, an hyponym. Finally, we have a semantic distance between two ImageNet classes that is the length of the shortest path between the two classes in the minimum spanning tree. This induces a matrix $$D$$ of size $$(1000, 1000)$$ that is symetric and with zero diagonal.

Given a feature $$d_k$$, we can then compute the average activation of this feature over each class. This gives a vector in $$\mathbb{R}^{1000}$$:


$$
\begin{equation*}
\mathrm{freq}(d_k)_i = \frac{1}{N_i}\sum_{j=1}^{N_i}f_k(\mathbf{x}_{ij})
\end{equation*}
$$


where $$N_i$$ is the number of images in class $$i$$ and $$f_k(\mathbf{x}_{ij})$$ is the activation of feature $$d_k$$ for image $$j$$ in class $$i$$. 

We propose the following metric to evaluate the semantic diameter of a feature $$d_k$$:

$$
\begin{equation*}
\mathrm{diameter}(d_k) = \frac{1}{\vert\vert\mathrm{freq}(d_k)\vert\vert_2^2}\mathrm{freq}(d_k)^\intercal D \mathrm{freq}(d_k)
\end{equation*}
$$

This measure how monosemantic a feature is with respect to semanticity of ImageNet classes. If a feature is activated by images of a single class, it will have a low score. If it is activated by a group of very close classes like different breeds of dogs, it will have a slightly higher score, etc...

This metric is far from perfect. Most notably for early layers, it is not clear that ImageNet classes are the best semantic description of the world. 

## 2. Initial Results

### 2.1 Monosemanticity of features

#### 2.1.1 Visual examination of features and neurons

In this section, we will present some of the learned features and try to interpret them. We will also look at neurons activations and compare them in term of mono/polysemanticity with the learned features. More precisely, for each feature (direction $$\mathbf{d}_k$$), we will compute the 100 images that activate it the most (maximize $$f_k$$). For each neuron (cartesian coordinate in $$\mathbb{R}^{n_classes}$$) we do the same and compute the 100 images that activate it the most. We then plot the top 25 images for each feature and each neuron. We then try to interpret the learned feature and the neuron. 

##### Layer 5b

This layer is the last layer before the classification head. It exhibits significant superposition. Yet, the fact that it is one of the last layer should mean that it might be easier for the auto-encoder to find mono-semantic features (a few linear layers after mixed 5b are then able to deconvolve 5b activations into logits with good precision).

Some of the learned features correspond directly to specific classes in ImageNet. For example, in run ft92b79c (5080 bottleneck neurons, $$\lambda=10^{-5}$$), feature 0 corresponds to the "mortarboard" class. The feature seems to have learned to detect the shape of a motarboard, as well as the presence of academic robes or diplomas. Surprisingly in the top 25 images, we also find an image of a mortar and one of a monitor. If a mortar has a similar shape, the monitor does not. THis is a clear example that our model isn't perfect. To save on compute ressources, we could not follow Anthropic's suggestion to train the auto-encoder "long after loss has plateaued". 

Other learned features do not correspond to a specific class but are visually very interpretable and monosemantic. For example, in the same run as above, feature 2 seems to descripe pointy metalic objects (screws, paintbrushs with a metalic guard, nails, corkscrews, revolver with or without bullets, drumsticks, shovel...).

Some feature are monosemantic in the animal kingdom. For example, in run ft92b79c, feature 10 correponds to insects/animal shells. In the top 25 images we recognize a lot of insects with articulated shells but also a species of turtle (terrapin), a conch, trilobite and a few snakes. Interestingly, a rusty radiator also found his way in the top 25 images (but it shares some visual similarity with an articulated shell).

Other feature include people wearing pink, something that looks like a train/truck, the content of a pot (food), very dark animals, a group of speedometers, compass and other -meters, things with thin stripes of yellow, black and white (from a hornet to a snake to a hockey player)...

Conversely, when looking at the feature that gets most activated by dogs in general and then looking at the top 25 images, we find only dogs. That was expected but one should note that the feature seems even more precise with most dogs having hanging ears and a fur color that is either black, white, brown or a mix of these colors. Other dog features also seem to have a semantic meaning that is more precise than just "dog". The second feature most activated by dogs seems to describe dogs with short hair, colors that are grey, white or at least pale in some way. The third feature is about dogs with abudant hairs that obstruct their face (tibetan terrier, briards...). The fourth and fifth are respectively about EntleBucher and flat and curly-coated retrivers. 

On the contrary, when looking at random neurons. Mixed5b seems to exihibit even more superposition than mixed 4a. Perhaps this is due to the fact that Mixed4a being present earlier might correspond to visually more monosemantic images. In appendix, we provide examples of top 25 images for random neurons in mixed5b.

When looking at 100 random features, 46 were found to be interpretable and monoseantic, 29 interpretable but with some superposition (less than 20%) or doubt with respect to the interpretation, 17 were somewhat interpretable but with significant superposition leading to significant doubt on the interpretation and 8 were not interpretable at all. Please note this is a very subjective evaluation.

When doing the same experiment with 100 random neurons, 2 were found to be interpretable and monosemantic, 15 interpretable but with some superposition (less than 20%) or doubt with respect to the interpretation, 48 were exhibited significant superposition leading to significant doubt on the interpretation and 35 were not interpretable at all. Again, if this was a trial between features and neurons, I would be the judge (in chosing which experiment to "allow"), the jury (in evaluating the results) and the executioner (in writing this report).


#### 2.1.2 Feature visualization

One of the next step will be to apply standard feature visualization techniques to the activations of our encoder. It will be interesting to see if interpretable features are reflected in the fabricated images and if clarity of said images is improved.

### 2.2 Sparsity, how sparse are learned features, what does a feature look like?

On average, for layer mixed 5b, a bottleneck of size 5080 and $$\lambda = 10^{-5}$$, an image activates 15 features, with a standard deviation of 6.5 features. As expected, the learned features are much sparser than neurons. On average, 764 neurons are activated by an image, with a standard deviation of 73 neurons. In the appendix, we provide distributions for both neurons and features activations.

A natural question then is to ask what a feature looks like. Like Anthropic, we compute the proportion of the squared norm of the feature (direction $$\mathbf{d}_k$$) that is explained by the top neuron (coordinate of $$\mathbf{d}_k$$ with highest square value) versus the next 9 neurons. As with their work on language models, we find that a majority of features are dense in the neuron basis with only few features being explained very well by the first 10 neurons (let alone the first neuron). See the appendix for a scatter plot of the proportion of the squared norm of the feature explained by the top neuron versus the next 9 neurons.

### 2.3 Metric evaluation

The learned feature also exihibted a significantly smaller semantic diameter than neurons. For the layer mixed5b and the run ft92b79c (5080 bottleneck neurons, $$\lambda=10^{-5}$$), the average semantic diameter of a neuron was $$7644 \pm 24$$ while the average diameter of a feature was $$394 \pm 54$$. We provide in appendix a histogram of the semantic diameter of neurons and features for the same run. This is encouraging, showing that our metric is in agreement with our visual examination of features and neurons.

## 4. Discussion and Perspectives

### 4.3 TODOs and open questions

CLIP metric

## Conclusion

## References

## Appendix

### Appendix A: Visual examination of neurons and features

We provide here a few examples of top 25 images for features and neurons. We also provide our interpretation of some features.

First, here are some examples of top 25 images for random features learned by our auto-encoder on mixed 5b activations:



### Appendix B: Semantic Diameter

Here is a plot of the matrix $$D$$:

<figure style="text-align: center;">
  <img style="display: inline-block;" width="300" height="300" src="/docs/D.png">
  <figcaption style="display: block;">Figure App.B.1: The matrix D induced by ImageNet hierarchy</figcaption>
</figure>
One can clearly see some structure, with some large blocks of classes that are close to each other.

And now a histogram of the semantic diameter of neurons and features for the run ft92b79c (5080 bottleneck neurons, $$\lambda=10^{-5}$$, layer mixed5b):

<figure style="text-align: center;">
  <img style="display: inline-block;" width="400" height="300" src="/docs/semantic_diameter.png">
  <figcaption style="display: block;">Figure App.B.2: The matrix D induced by ImageNet hierarchy</figcaption>
</figure>

The reader should note that the semantic diameter of neurons is much larger than the semantic diameter of features.

### Appendix C: Sparsity

In this section, we provide histograms of the number of neurons and features activated by an image for the run ft92b79c (5080 bottleneck neurons, $$\lambda=10^{-5}$$, layer mixed5b):

<figure style="text-align: center;">
  <img style="display: inline-block;" width="400" height="300" src="/docs/num_act_features_neurons.png">
  <figcaption style="display: block;">Figure App.B.2: The matrix D induced by ImageNet hierarchy</figcaption>
</figure>
