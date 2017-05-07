from glob import glob
from skimage import io, color
import sys
import os
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from lxml import etree
from sklearn.svm import LinearSVC

NEIGHBORHOOD_SIZE = 5

def extract_with_padding(datapoints, i, j):
    i_min = max(i - NEIGHBORHOOD_SIZE, 0)
    i_max = min(i + NEIGHBORHOOD_SIZE, datapoints.shape[0])
    j_min = max(j - NEIGHBORHOOD_SIZE, 0)
    j_max = min(j + NEIGHBORHOOD_SIZE, datapoints.shape[1])
    pad_i = (max(NEIGHBORHOOD_SIZE - i, 0), max(i + NEIGHBORHOOD_SIZE - datapoints.shape[0], 0))
    pad_j = (max(NEIGHBORHOOD_SIZE - j, 0), max(j + NEIGHBORHOOD_SIZE - datapoints.shape[1], 0))
    x = datapoints[i_min:i_max, j_min:j_max]
    x = np.pad(x, [pad_i, pad_j, (0, 0)], mode='constant', constant_values=(0, 0))
    x = np.reshape(x, (-1))

    return x


if __name__ == '__main__':
    files = glob(os.path.join(sys.argv[1], '*.xml'))
    features = []
    labels = []
    for file in files:
        tree = etree.parse(file)
        fname = os.path.basename(file).strip('.xml') + '.png'
        img = io.imread(os.path.join(sys.argv[2], fname))
        img_yuv = color.rgb2yuv(img)
        areas = tree.xpath('.//object')

        for area in areas:
            label = area.xpath('./name/text()')[0]
            xmin = int(area.xpath('./bndbox/xmin/text()')[0])
            xmax = int(area.xpath('./bndbox/xmax/text()')[0])
            ymin = int(area.xpath('./bndbox/ymin/text()')[0])
            ymax = int(area.xpath('./bndbox/ymax/text()')[0])
            datapoints = img_yuv[ymin:ymax, xmin:xmax]
            for i in range(datapoints.shape[0]):
                for j in range(datapoints.shape[1]):
                    x = extract_with_padding(datapoints, i, j)
                    features.append(x)
                    labels.append(label)

    labelencoder = LabelEncoder()
    y = labelencoder.fit_transform(labels)
    print(labelencoder.classes_)
    X = np.vstack(features)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    svc = LinearSVC()
    svc.fit(X_train, y_train)
    print(classification_report(y_test, svc.predict(X_test)))
    joblib.dump(svc, 'svm.pkl')
    joblib.dump(labelencoder, 'labelencoder.pkl')

