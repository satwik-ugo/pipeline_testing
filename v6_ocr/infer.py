#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from yolov6.core.inferer import Inferer
from tqdm import tqdm
import numpy as np
import PIL, io, cv2, torch, time, sys, os, argparse
from character_recognition import CharacterRecognizer
recognizer = CharacterRecognizer(verbose=False)

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='YOLOv6 PyTorch Inference.', add_help=add_help)
    parser.add_argument('--show', action='store_true', help='enable view image')
    parser.add_argument('--device', default='cpu', help='device to run our model i.e. 0 or 0,1,2,3 or cpu.')
    args = parser.parse_args()
    return args

# @torch.no_grad()
# def run(
#         show=False,
#         device='',
#         source="test.jpg",
#         conf_thres=0.5,
#         iou_thres=0.8,
#         agnostic_nms=False,
#         ):

#     # Inference
#     inferer = Inferer(source, device)
#     t1 = time.time()
#     det = inferer.infer(conf_thres, iou_thres, agnostic_nms, show)

#     t2 = time.time()

#     print("Elapse_time == ", t2-t1)

def button_candidates(boxes, scores, image):

    button_scores = []  # stores the score of each button (confidence)
    button_patches = []  # stores the cropped image that encloses the button
    button_positions = []  # stores the coordinates of the bounding box on buttons

    for box, score in zip(boxes, scores):
        if score < 0.5:
            continue

        y_min = int(box[0])
        x_min = int(box[1]) 
        y_max = int(box[2])
        x_max = int(box[3])

        if x_min < 0 or y_min < 0:
            continue
        button_patch = image[y_min: y_max, x_min: x_max]
        button_patch = cv2.resize(button_patch, (180, 180))

        button_scores.append(score)
        button_patches.append(button_patch)
        button_positions.append([x_min, y_min, x_max, y_max])
    return button_patches, button_positions, button_scores

def main(args):

    img_paths = []
    test_dir = "../test_images/"
    for file_name in os.listdir(test_dir):
        file_path = test_dir + file_name
        img_paths.append(file_path)
    
    st = time.time()
    for path in tqdm(img_paths[0:100]):

        with open(file_path, 'rb') as f:
            img_np = np.asarray(PIL.Image.open(io.BytesIO(f.read())))
        inferer = Inferer(path, device='')
        det = inferer.infer(conf_thres = 0.5, iou_thres=0.8,agnostic_nms=False,view_img=False)
    
        det = det.tolist()
        boxes = [row[:4] for row in det]
        scores = [row[4] for row in det]

        button_patches, button_positions, _ = button_candidates(
            boxes, scores, img_np)

        for button_img in button_patches:
            # get button text and button_score for each of the images in button_patches
            button_text, button_score, _ = recognizer.predict(button_img)
            # print(button_text)

    end = time.time()
    time_taken = end-st
    print(f"Time taken for inferring 100 images with Yolov6+OCR is {time_taken}")


    # run(**vars(args))

if __name__ == "__main__":
    args = get_args_parser()
    main(args)
