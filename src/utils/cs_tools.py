import numpy as np

def reduce_classes(gt : np.ndarray, class_id: int = 20):
    """
    Reduce a multi-class ground truth to a binary ground truth, where given main_class gets label 1 and everything else is background with label 0
    Parameters:
    gt(np.ndarray): Ground Truth to transform, should be in labelmap shape (1, H, W)
    class_id(int): Label of class to retain
    """
    if isinstance(gt, np.ndarray):
        if gt.ndim == 2:
            gt = (gt == class_id)
            gt.dtype = np.uint8
        else:
            raise Exception(f"gt should be of shape (H, W), but has shape {gt.shape}")
    else:
        gt = np.array(gt)
        if gt.ndim == 2:
            gt = (gt == class_id)
            gt.dtype = np.uint8
        else: 
            raise Exception(f"gt should be of shape (H, W), but has shape {gt.shape}")
    return gt