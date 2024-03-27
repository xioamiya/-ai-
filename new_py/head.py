import mediapipe as mp


# Load the input image from an image file.
mp_image = mp.Image.create_from_file('img/1.jpg')

# Load the input image from a numpy array.
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=mp_image)
