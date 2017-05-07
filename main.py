import cv2
import os
from glob import glob

import sys
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from skimage import color
from skimage import io
from sklearn.externals import joblib
from tqdm import tqdm

from gen_data import extract_with_padding

Y = 0
U = 1
V = 2

COOLDOWN = 10

RELEVANT_TRANSITIONS = [
    [('blue', 'yellow'), ('yellow', 'white'), ('white', 'green')],
    [('yellow', 'blue'), ('blue', 'white'), ('white', 'green')],

    [('yellow', 'red'), ('red', 'white'), ('white', 'green')],
    [('red', 'yellow'), ('yellow', 'white'), ('white', 'green')],

    [('blue', 'red'), ('red', 'white'), ('white', 'green')],
    [('red', 'blue'), ('blue', 'white'), ('white', 'green')],
]

svm = joblib.load('svm.pkl')
labelencoder = joblib.load('labelencoder.pkl')

def determine_colors(yuv):
    colors = labelencoder.inverse_transform(svm.predict(yuv))
    return colors

def scan_y(img_yuv, x):
    color_memory = defaultdict(int)
    color_transitions = []
    features = []
    for y in range(img_yuv.shape[0]):
        features.append(extract_with_padding(img_yuv, y, x))

    colors = determine_colors(features)
    for y, color in enumerate(colors):
        if color != 'other':
            color_memory[color] = COOLDOWN

            for k in color_memory:
                if color != k and color_memory[k] > 0 :
                    if (len(color_transitions) == 0) or (color_transitions[-1][:2] != (k, color)):
                        color_transitions.append((k, color, y))

        for k in color_memory:
            color_memory[k] -= 1

    return color_transitions, colors



images = glob(os.path.join(sys.argv[1], '*'))
with open('results.csv','w') as f:
    for img_fname in tqdm(images):
        img = io.imread(img_fname)
        img_yuv = color.rgb2yuv(img)
        bounding_boxes = []

        xs = []
        ys = []

        for x in range(img_yuv.shape[1]):
            found_pylon = False
            color_transitions, colors = scan_y(img_yuv, x)
            for relevant_transitions in RELEVANT_TRANSITIONS:
                relevant_transition_idx = 0
                for transition_idx, transition in enumerate(color_transitions):
                    relevant_transition = relevant_transitions[relevant_transition_idx]

                    if transition[:2] == relevant_transition:
                        relevant_transition_idx += 1
                    else:
                        if transition[:2] == relevant_transitions[0]:
                            relevant_transition_idx = 1
                        else:
                            relevant_transition_idx = 0

                    if relevant_transition_idx == len(relevant_transitions):
                        if transition_idx - len(relevant_transitions) >= 0:
                            y_min = color_transitions[transition_idx - len(relevant_transitions)][2]
                        else:
                            y_min = color_transitions[transition_idx - len(relevant_transitions)+1][2]
                        y_max = transition[2]

                        xs += [x]
                        ys += [y_min, y_max]
                        found_pylon = True
                        break
            if not found_pylon and len(xs) > 0:
                bounding_boxes.append(((np.min(xs), np.min(ys)), (np.max(xs), np.max(ys))))
                xs = []
                ys = []

        f.write('{},{}'.format(img_fname, len(bounding_boxes)))
        for box in bounding_boxes:
            cv2.rectangle(img, box[0], box[1], (255,0,0), 3)
            f.write(',{}, {}, {}, {}'.format(box[0][0], box[0][1], box[1][0], box[1][1]))
        f.write('\n')
        f.flush()
        test_fname = 'data/test/{}'.format(os.path.basename(img_fname))
        cv2.imwrite(test_fname, img)




