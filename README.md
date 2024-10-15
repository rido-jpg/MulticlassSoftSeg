# MulticlassSoftSeg

Trying to extend the approach explained in the [SoftSeg paper](https://arxiv.org/pdf/2011.09041.pdf) to a multi-class segmentation scenario.
The goal is to diminish the effects of the Partial Volume Effect in medical images with low resolution by using "soft" segmentation masks. These have soft transitions between the different classes inside the segmentation mask (ground truth) and allow to encode uncertainties. This also allows to encode expert annotations with uncertainty levels and use these for training models.
