import cv2
import numpy as np

def show_rgb_channels_single_canvas(image_path):
    """
    Loads an image and displays the individual grayscale versions
    of its Red, Green, and Blue channels on a single canvas.

    Args:
        image_path (str): The path to the image file.
    """
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    height, width, _ = img.shape

    b, g, r = cv2.split(img)

    # Corrected: Create a 2D canvas to match the 2D channel arrays
    # No need for the '1' at the end.
    canvas = np.zeros((height, width * 3 + 20), dtype=np.uint8)

    # Place the Red, Green, and Blue channel images on the canvas
    canvas[:, :width] = r
    canvas[:, width + 10 : width * 2 + 10] = g
    canvas[:, width * 2 + 20 : width * 3 + 20] = b

    # Display the combined image
    cv2.imshow('RGB Channels on One Canvas (R-G-B)', canvas)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Use the function with your image file
show_rgb_channels_single_canvas('data/raw_120_forRGBtest/image.png')