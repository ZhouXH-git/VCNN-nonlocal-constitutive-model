# VCNN - Vector Cloud Neural Network
The vector-cloud neural network (VCNN) is a promising tool not only as nonlocal constitutive models and but also as general surrogate models for PDEs on irregular domains. It is a frame-independent neural network, strictly invariant both to coordinate translation and rotation and to the ordering of points in the cloud. The network can deal with any number of arbitrarily arranged grid points and thus is suitable for unstructured meshes in fluid simulations.

Co-developed by Dr. Heng Xiao's group at Virginia Tech: [Data-Enabled Computational Mechanics Laboratory at Virgnia Tech](https://www.aoe.vt.edu/people/faculty/xiaoheng/personal-page.html) and Dr. Jiequn Han at Flatiron Institute: [Center for Computational Mathematics.](https://www.simonsfoundation.org/people/jiequn-han/)

![image](https://github.com/xuhuizhou-vt/VCNN-nonlocal-constitutive-model/blob/master/figs/schematic_workflow.png)

##### Schematic of the frame-independent, permutation-invariant vector-cloud neural network for nonlocal constitutive modeling, showing a mapping $\cQ \mapsto \tau$ from a cloud (left oval) of feature vectors $\cQ = \vstack{\bq_1^\top, \bq_2^\top, \ldots}$ to the closure variable $\tau$ (center of right oval). We construct the mapping by starting with two simultaneous operations: (i) extract pairwise inner-product to obtain rotational invariant features $\cD'_{ii'} = \bq_i^\top \bq_{i'}$ and (ii) map the scalar quantities $\bw$ in each vector $\bq$ in the cloud through an embedding network to form a permutational invariant basis $\cG$, which also inherits its rotational invariance from input~$\bw$. Then, we project $\cD'$ onto basis $\cG$ (not necessarily orthogonal) to produce final feature matrix $\cD$, which is invariant to frame and permutation. Finally, we fit a neural network to map features $\cD$ to closure variable $\tau$.

This repository contains the code and data for the following paper(s):

*   X-H. Zhou, J.Q. Han, and H. Xiao. Frame-independent vector-cloud neural network for nonlocal constitutive modelling on arbitrary grids. *Computer Methods in Applied Mechanics and Engineering*. In Press, 2021. Also available at arXiv: [https://arxiv.org/abs/2103.06685](https://arxiv.org/abs/2103.06685)

The current work is about scalar transport PDE. We are working on extending the VCNN to tensor transport PDE.

Contributors:
-------------
* Xu-Hui Zhou
* Jiequn Han
* Heng Xiao

Contact: Xu-Hui Zhou
