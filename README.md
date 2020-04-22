## Spatio-temporal activity sequence

Author: Sebastian Spreizer, Ad Aertsen, Arvind Kumar

#### Abstract

Spatio-temporal sequences of neuronal activity are observed in many brain regions in a variety of tasks and are thought to form the basis of meaningful behavior. However, mechanisms by which a neuronal network can generate spatio-temporal activity sequences have remained obscure. Existing models are biologically untenable because they either require manual embedding of a feedforward network within a random network or supervised learning to train the connectivity of a network to generate sequences. Here, we propose a biologically plausible, generative rule to create spatio-temporal activity sequences in a network of spiking neurons with distance-dependent connectivity. We show that the emergence of spatio-temporal activity sequences requires: (1) individual neurons preferentially project a small fraction of their axons in a specific direction, and (2) the preferential projection direction of neighboring neurons is similar. Thus, an anisotropic but correlated connectivity of neuron groups suffices to generate spatio-temporal activity sequences in an otherwise random neuronal network model.

#### Requirements

Check the requirements for the simulation in the file `./singularity/py3_activity_sequence.sif`.

Here, I make a list of requirements:

 - jupyter
 - matplotlib
 - NEST Simulator (v2.18.0)
 - noise
 - numpy
 - scikit-learn
 - scipy


#### Usage

I have prepared script for Jupyter notebook that you are able to perform simulation generating STAS in EI-network or in I-network model.

First, go to `./notebook` folder and then start Jupyter Notebook:

```bash
jupyter notebook
```


#### Reference

 - [Spreizer, S., Aertsen, A., & Kumar, A. (2019). From space to time: Spatial inhomogeneities lead to the emergence of spatiotemporal sequences in spiking neuronal networks. PLOS Computational Biology, 15(10), e1007432. https://doi.org/10.1371/journal.pcbi.1007432](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007432)
