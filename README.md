# MulticlassSoftSeg

Extending the approach explained in the [SoftSeg paper](https://arxiv.org/pdf/2011.09041.pdf) to a 3D multi-class segmentation scenario, as part of my master's thesis.

Citing from the abstract of my thesis:

Traditional segmentation approaches rely on _hard_ reference annotations, which fail to
account for the partial volume effect (PVE) and lead to significant global and local volume information loss.
This thesis investigates the potential of using _soft_ reference annotations derived from binarized reference
annotations through Gaussian filtering to address these limitations. Using the 3D multi-class Brain Tumor
Segmentation (BraTS) dataset, this study compares the performance of models trained with _soft_ anno-
tations against traditional _hard_-label-based methods. Results demonstrate that models trained on artificially softened anno-
tations are able to recover partial volume information lost during binarization, enabling better prediction
of mixed tissue compositions while maintaining competitive performance in _hard_-segmentation tasks. Fur-
ther, they showed an improved ability to recognize and predict larger-scale tumor instances.

