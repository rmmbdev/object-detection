import numpy as np
import cv2
import tensorflow as tf


def multi_scale_edge_detection(image):
    edges = []
    for scale in range(1, 5):
        resized_image = cv2.resize(image, dsize=(0, 0), fx=1 / scale, fy=1 / scale)
        edge_map = cv2.Canny(resized_image, 50, 150)
        edges.append(edge_map)

    combined_edge_map = np.zeros_like(edges[0])
    for edge_map in edges:
        combined_edge_map += edge_map

    combined_edge_map = cv2.normalize(combined_edge_map, None, 0, 255, cv2.NORM_MINMAX)
    return combined_edge_map


def cnn_horizon_detection(edge_map):
    model = tf.keras.models.load_model('horizon_detection_model.h5')
    edge_map = np.expand_dims(edge_map, axis=2)
    edge_map = np.expand_dims(edge_map, axis=0)
    prediction = model.predict(edge_map)
    horizon_line = prediction[0]
    return horizon_line


def linear_curve_fitting(horizon_line):
    horizon_line = horizon_line.ravel()
    y, x = horizon_line.nonzero()
    m, c = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y, rcond=None)[0]
    return m, c


def median_filtering(horizon_line):
    horizon_line = np.median(horizon_line, axis=1)
    return horizon_line


def horizon_detection(image):
    edge_map = multi_scale_edge_detection(image)
    horizon_line = cnn_horizon_detection(edge_map)
    horizon_line = linear_curve_fitting(horizon_line)
    horizon_line = median_filtering(horizon_line)
    return horizon_line


if __name__ == '__main__':
    image = cv2.imread('image.jpg')
    horizon_line = horizon_detection(image)
    cv2.line(image, (0, horizon_line[1]), (image.shape[1], horizon_line[1]), (0, 0, 255), 2)
    cv2.imshow('Horizon detection', image)
    cv2.waitKey(0)
