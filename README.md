## Spatio-temporal activity sequence

Author: Sebastian Spreizer, Ad Aertsen, Arvind Kumar \
Source code: https://github.com/babsey/spatio-temporal-activity-sequence

#### Abstract

Spatio-temporal sequences of neuronal activity are observed in many brain regions in a variety of tasks and are thought to form the basis of meaningful behavior. However, mechanisms by which a neuronal network can generate spatio-temporal activity sequences have remained obscure. Existing models are biologically untenable because they either require manual embedding of a feedforward network within a random network or supervised learning to train the connectivity of a network to generate sequences. Here, we propose a biologically plausible, generative rule to create spatio-temporal activity sequences in a network of spiking neurons with distance-dependent connectivity. We show that the emergence of spatio-temporal activity sequences requires: (1) individual neurons preferentially project a small fraction of their axons in a specific direction, and (2) the preferential projection direction of neighboring neurons is similar. Thus, an anisotropic but correlated connectivity of neuron groups suffices to generate spatio-temporal activity sequences in an otherwise random neuronal network model.

#### Description of the connectivity algorithm

For the network construction we used own algorithm for the connectivity in space. We consider the network space in grid pattern which is based on neuron IDs. The algorithm takes neuron IDs the size of the network space and the amount of connections. Then it calculates for each neuron the targets for the connection.
To break the symmetry, we shift target ids towards a direction with the landscape of the Perlin noise.

You can find codes of:
 - LCRN connectivity in `./scripts/lib/lcrn_network.py`
 - asymmetric connection approach in `./scripts/simulation/sequence_EI_networks.py`

### Usage

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


#### Jupyter notebook

I have prepared script for Jupyter notebook that you are able to perform simulation generating STAS in EI-network or in I-network model.

First, go to `./notebook` folder and then start Jupyter Notebook:

```bash
jupyter notebook
```

### Reference

 - [Spreizer, S., Aertsen, A., & Kumar, A. (2019). From space to time: Spatial inhomogeneities lead to the emergence of spatiotemporal sequences in spiking neuronal networks. PLOS Computational Biology, 15(10), e1007432. https://doi.org/10.1371/journal.pcbi.1007432](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007432)
