import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist


def removeNeutralColors(clt):
    white = [255,255,255]
    black = [0,0,0]
    grey = [128,128,128]
    white = clt.predict(white)
    black = clt.predict(black)
    grey = clt.predict(grey)
    return[black.astype("uint8").tolist()[0], white.astype("uint8").tolist()[0], grey.astype("uint8").tolist()[0]]

def dominantColor(hist, centroids):
    max = 0
    clr = 0
    newCentroid = []
    newHist = []
    index = removeNeutralColors(clt)
    print(index)
    for i in range(4):
        if(i not in index):
            newCentroid.append(centroids[i])
            newHist.append(hist[i])
    if(centroids != []):
        for (percent, color) in zip(newHist, newCentroid):
                print(color)
                if(percent>max):
                    max = percent
                    clr = color


    # return the bar chart
    return clr.astype("uint8").tolist()


img = cv2.imread("2.jpeg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = img.reshape((img.shape[0] * img.shape[1],3)) #represent as row*column,channel number
clt = KMeans(n_clusters=4) #cluster number
clt.fit(img)



hist = find_histogram(clt)
bar = dominantColor(hist, clt.cluster_centers_)

print(bar)
