import cv2
import numpy as np


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
