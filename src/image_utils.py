import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = '/code/'


def draw_rectangle_with_text_wrt_hw(image, x, y, w, h, text):
    return draw_rectangle_with_text_wrt_points(image, x, y, x + w, y + h, text)


def draw_rectangle_with_text_wrt_points(image, x1, y1, x2, y2, text):
    # Rectangle parameters
    rectangle_color = (255, 0, 0)  # Blue color in BGR format (Blue, Green, Red)
    rectangle_thickness = 2  # Thickness of the rectangle's border (in pixels)
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw the blue rectangle on the image
    image = cv2.rectangle(image, (x1, y1), (x2, y2), rectangle_color, rectangle_thickness)

    # Text parameters
    font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
    font_scale = 1  # Font scale
    font_color = (0, 0, 0)  # Black color for the text in BGR format
    font_thickness = 2  # Thickness of the font (in pixels)

    # Calculate the position to center the text within the rectangle
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_x = x1
    text_y = y1

    # Draw the text on the image
    image = cv2.putText(image, text, (text_x, text_y), font, font_scale, font_color, font_thickness)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def draw_point(image, x, y):
    point_coordinates = (x, y)
    color = (0, 0, 255)  # BGR color, so (0, 0, 255) is red
    thickness = -1  # Thickness -1 fills the circle
    radius = 5  # Radius of the circle

    cv2.circle(image, point_coordinates, radius, color, thickness)

    return image


def save_image(image):
    fig, axs = plt.subplots(1, 1)
    axs.imshow(image)
    axs.set_title('Original')
    width_inches = 9.03  # Adjust as needed
    height_inches = 6.01  # Adjust as needed
    fig.set_size_inches(width_inches, height_inches)

    # buffer = BytesIO()
    fig.savefig(
        os.path.join(ROOT, "results", f"tmp.png"),
        bbox_inches='tight',
        dpi=300
    )
    plt.close(fig)
