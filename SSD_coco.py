import glob
import os
import time

import torch
from PIL import Image
from vizer.draw import draw_boxes

from ssd.config import cfg
from ssd.data.datasets import COCODataset, VOCDataset
import argparse
import numpy as np

from ssd.data.transforms import build_transforms
from ssd.modeling.detector import build_detection_model
from ssd.utils import mkdir
from ssd.utils.checkpoint import CheckPointer



@torch.no_grad()
def run_demo(cfg, ckpt, score_threshold, image_path, dataset_type):
	class_names = COCODataset.class_names
	device = torch.device('cpu')
	model = build_detection_model(cfg)
	model = model.to('cpu')
	checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
	checkpointer.load(ckpt, use_latest=ckpt is None)
	weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
	print('Loaded weights from {}'.format(weight_file))

	cpu_device = torch.device("cpu")
	transforms = build_transforms(cfg, is_train=False)
	model.eval()
	image_name = os.path.basename(image_path)
	image = np.array(Image.open(image_path).convert("RGB"))
	height, width = image.shape[:2]
	images = transforms(image)[0].unsqueeze(0)

	start = time.time()
	load_time = time.time() - start
	result = model(images.to('cpu'))[0]
	inference_time = time.time() - start

	result = result.resize((width, height)).to(cpu_device).numpy()
	boxes, labels, scores = result['boxes'], result['labels'], result['scores']
	indices = scores > score_threshold
	boxes = boxes[indices]
	labels = labels[indices]
	scores = scores[indices]
	print("There are {} number of objects detected ".format(len(boxes)))
	print("Total time taken to detect the {} number of objects is {}".format(len(boxes),inference_time))
	for i,j in enumerate(labels):
		print(class_names[j])
		print(boxes[i],type(boxes[i]))
		box = list(boxes[i])
		print(box)

if __name__=="__main__":
	cfg.merge_from_file('configs/vgg_ssd300_coco_trainval35k.yaml')
	cfg.freeze()
	run_demo(cfg=cfg,
             ckpt="vgg_ssd300_coco_trainval35k.pth",
             score_threshold=0.85,
             image_path="demo\\cars1.jpg",
             dataset_type="coco")