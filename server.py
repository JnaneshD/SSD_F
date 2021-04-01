from flask import Flask, render_template, request, jsonify
from werkzeug.utils  import secure_filename

from flask.ext.cors import CORS, cross_origin
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'

UPLOAD_FOLDER = 'C:/Users/acer/Desktop/SSD/uploaded_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
import os
import sys
import base64
import cv2
import json
sys.path.append("C:/Users/acer/Desktop/SSD")
import glob
import os
import time
import torch
from PIL import Image
from ssd.config import cfg
from ssd.data.datasets import COCODataset, VOCDataset
import numpy as np
from ssd.data.transforms import build_transforms
from ssd.modeling.detector import build_detection_model
from ssd.utils import mkdir
from ssd.utils.checkpoint import CheckPointer
cfg.merge_from_file('configs/vgg_ssd512_coco_trainval35k.yaml')
cfg.freeze()
'''
Load the SSD weights 
'''
ckpt="vgg_ssd512_coco_trainval35k.pth"
class_names = COCODataset.class_names
device = torch.device('cpu')
model = build_detection_model(cfg)
model = model.to('cpu')
checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
checkpointer.load(ckpt, use_latest=ckpt is None)
weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
print('Loaded weights from {}'.format(weight_file))


def delete_github():
	os.chdir("uploaded_images")
	os.system("del * /F /Q")
	os.system("git rm -r -all")
	os.system("git commit -am '"+str(time.time())+"'")
	os.system("git push -u origin main")
	os.chdir("..")
def push_git():
	os.chdir("uploaded_images")
	os.system("git add .")
	os.system("git commit -m '"+str(time.time())+"'")
	os.system("git push -u origin main")
	os.chdir("..")

@torch.no_grad()
def run_demo(cfg, ckpt, score_threshold, image_path, dataset_type):
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
	response = {}
	print(boxes,type(boxes))
	img = cv2.imread("uploaded_images/imageToSave.png")
	for i,J in enumerate(labels):
		box = list(boxes[i])
		cropped = img[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
		#cv2.imshow("cropped",cropped)
		#cv2.waitKey(0)
		time_cur = str(time.time())
		cv2.imwrite("uploaded_images/"+class_names[J]+time_cur+".png",cropped)
		response[class_names[J]+time_cur] = str(box[0])+','+str(box[1])+','+str(box[2])+','+str(box[3])
	push_git()
	return response

cors = CORS(app, resources={r"/detect": {"origins": "*"}})


def allowed_file(filename):
	return '.' in filename and \
			filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/app",methods=['GET'])
def dead():
	return "boo"



@app.route('/detect', methods=['POST'])
def upload_file():
	coordinates = None
	new_dict = {}
	if request.method == 'POST':
		# check if the post request has the file part
		#if 'file' not in request.files:
		#	flash('No file part')
		#	return redirect(request.url)
		#file = request.files['file']
		files = request.form.get("file")
		files = files.split(",")[-1]
		#delete_github()
		with open("uploaded_images/imageToSave.png", "wb") as fh:
		    fh.write(base64.b64decode(files))
		file_path = os.path.join(app.config['UPLOAD_FOLDER'], '')
		file_path+="imageTosave.png"
		
		result = run_demo(cfg=cfg,
             ckpt="vgg_ssd300_coco_trainval35k.pth",
             score_threshold=0.4,
             image_path=file_path,
             dataset_type="coco")
		print(result)
		
		
		#ssd_detect = Object()
		#cooridinates = ssd_detect.detect_ssd(file_path)
		#kk = jsonify(cooridinates)
		#print(cooridinates,type(cooridinates))
#			res = json.dumps(new_dict)
	return jsonify(result)

	
if __name__ == '__main__':
	app.run(debug = True)
