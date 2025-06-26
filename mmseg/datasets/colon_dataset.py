import os.path as osp
import numpy as np
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class ColonDataset(CustomDataset):
    """Colon polyp segmentation dataset.
    
    In segmentation map annotation for ColonDataset, 0 stands for background, 
    which is included in 2 categories. The ``img_suffix`` is fixed to '.png' 
    and ``seg_map_suffix`` is fixed to '.png'.
    """
    
    CLASSES = ('background', 'polyp')
    PALETTE = [[0, 0, 0], [255, 255, 255]]
    
    def __init__(self, **kwargs):
        super(ColonDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        
    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, 
                         split):
        """Load annotation from directory.
        
        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str): Path to annotation directory.
            seg_map_suffix (str): Suffix of segmentation maps.
            split (str): Split txt file. If split is specified, only file names
                in the splits will be loaded. Otherwise, all images in img_dir/ann_dir
                will be loaded. Default: None
        
        Returns:
            list[dict]: All image info of dataset.
        """
        
        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            # Load all images
            import os
            img_files = []
            if osp.exists(img_dir):
                for file in os.listdir(img_dir):
                    if file.endswith(img_suffix):
                        img_files.append(file)
            
            img_files.sort()
            
            for img_file in img_files:
                img_name = osp.splitext(img_file)[0]
                img_info = dict(filename=img_file)
                if ann_dir is not None and osp.exists(ann_dir):
                    seg_map = img_name + seg_map_suffix
                    seg_map_path = osp.join(ann_dir, seg_map)
                    if osp.exists(seg_map_path):
                        img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
                
        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos
        
    def get_gt_seg_maps(self):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for img_info in self.img_infos:
            seg_map = osp.join(self.ann_dir, img_info['ann']['seg_map'])
            gt_seg_map = np.array(Image.open(seg_map), dtype=np.uint8)
            # Convert to binary mask (0: background, 1: polyp)
            gt_seg_map = (gt_seg_map > 127).astype(np.uint8)
            gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps
        
    def pre_eval(self, preds, indices):
        """Collect eval result from each iteration.
        
        Args:
            preds (list[torch.Tensor] | torch.Tensor): the segmentation logit
                after argmax, shape (N, H, W).
            indices (list[int] | int): the prediction related ground truth
                indices.
                
        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction,
                area_ground_truth).
        """
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        pre_eval_results = []
        for pred, index in zip(preds, indices):
            seg_map = osp.join(self.ann_dir, self.img_infos[index]['ann']['seg_map'])
            gt_seg_map = np.array(Image.open(seg_map), dtype=np.uint8)
            # Convert to binary mask
            gt_seg_map = (gt_seg_map > 127).astype(np.uint8)
            
            pred = pred.cpu().numpy().astype(np.uint8)
            
            # Calculate intersection and union for Dice and IoU
            intersect = pred * gt_seg_map
            area_intersect = np.histogram(
                intersect, bins=np.arange(self.num_classes + 1))[0]
            area_pred_label = np.histogram(
                pred, bins=np.arange(self.num_classes + 1))[0]
            area_label = np.histogram(
                gt_seg_map, bins=np.arange(self.num_classes + 1))[0]
            area_union = area_pred_label + area_label - area_intersect
            
            pre_eval_results.append((area_intersect, area_union, 
                                   area_pred_label, area_label))
        return pre_eval_results
        
    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.
        
        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if palette is None:
            if self.PALETTE is None:
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
            else:
                palette = self.PALETTE

        return class_names, palette


from mmseg.utils import get_root_logger
from mmcv.utils import print_log
import mmcv 