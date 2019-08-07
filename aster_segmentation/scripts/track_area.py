import numpy as np
import skimage
import matplotlib.pyplot as plt
import imageio
from matplotlib.transforms import Bbox
import cv2
import csv



"""
This program outputs a graph of the asters area over time as well as saving it to a csv.

the make_movie() function saves a movie that has the outline of the region used for the area tracking



"""


movie_dir = "/Users/jackstellwagen/Desktop/FeatureNet_results/200um_preliminary/"
movie_name = "200_AR_0.75_annotated.avi"
outlined_movie_name = "200_AR_0.75_outlined.avi"
csv_name = "200_AR_0.75.csv"






cap = cv2.VideoCapture(movie_dir + movie_name)
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

coords = np.empty(shape=(frameWidth*frameHeight,2))
count = 0
for i in range(frameWidth):
    for j in range(frameHeight):
        coords[count][0] = i
        coords[count][1] = j
        count += 1
cap.release()

def track_area(dir, movie):
    cap = cv2.VideoCapture(dir + movie)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fc = 0
    ret = True
    areas = []
    while (fc < frameCount  and ret):
        ret, frame = cap.read()
        areas += [calculate_area(frame)]
        fc += 1
    cap.release()
    return areas


def calculate_area(im):
    aster = np.round(im / 100.0).astype("uint8")


    aster = (aster==1).astype("uint8")
    contours = skimage.measure.find_contours(aster[:,:,0], 0)
    if len(contours)==0:
        return 0

    inner_points = skimage.measure.points_in_poly(coords, contours[0])
    return coords[inner_points].shape[0]


def plot_area(area_arr):
    fig,ax = plt.subplots(1)
    ax.scatter(np.arange(len(area_arr)), area_arr)
    plt.show()






def make_movie(dir, movie, destination):



    cap = cv2.VideoCapture(dir + movie)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video = cv2.VideoWriter(dir + destination, 0, 10, (frameWidth,frameHeight))

    fc = 0
    ret = True
    areas = []
    while (fc < frameCount  and ret):
        ret, frame = cap.read()
        aster = (np.round(frame / 100.0).astype("uint8") == 1).astype("uint8")
        contours = skimage.measure.find_contours(aster[:,:,0], 0)
        if len(contours) != 0 :
            contours = contours[0].astype("uint8")
            frame[contours[:,0], contours[:,1]] = 255
        video.write(frame)
        fc += 1

    cap.release()
    video.release()


def write_csv(areas):
    with open(movie_dir + csv_name, 'w', newline='') as file:
         wr = csv.writer(file, quoting=csv.QUOTE_ALL)
         wr.writerow(areas)







if __name__ == '__main__':
    area = track_area(movie_dir, movie_name)
    plot_area(area)
    write_csv(area)
    make_movie(movie_dir, movie_name, outlined_movie_name)
