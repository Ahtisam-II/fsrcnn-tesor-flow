import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from google.colab import drive
import sys
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
time.sleep(5)
import os
drive.mount('/content/drive')

print(os.listdir("/content/drive/My Drive/Huzaifa_FSRCNN"))
custom_objects = {'mse': tf.keras.losses.MeanSquaredError()}

sys.path.append(os.path.abspath("/content/drive/My Drive/Huzaifa_FSRCNN"))
model_path = "/content/drive/My Drive/Huzaifa_FSRCNN/sr_cnn_model.h5"
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

model.summary()

def load_and_prepare_image(path, target_hr_size=(768, 768), scale=3):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hr = cv2.resize(img, target_hr_size, interpolation=cv2.INTER_CUBIC)
    h, w = target_hr_size
    lr = cv2.resize(hr, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
    lr_input = lr / 255.0
    hr = hr / 255.0
    return lr_input, hr, lr

def predict_sr(model, lr_image, target_size):
    input_tensor = np.expand_dims(lr_image, axis=0)
    predicted = model.predict(input_tensor)[0]
    predicted = np.clip(predicted, 0, 1)
    predicted_resized = cv2.resize(predicted, target_size, interpolation=cv2.INTER_CUBIC)
    return predicted_resized

def display_result(hr, predicted, lr_original):
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(lr_original / 255.0)  # Show LR as-is
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(predicted)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(hr)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

test_image_path = "/content/drive/My Drive/Huzaifa_FSRCNN/aunt_cool.png"

lr_img, hr_img, original_lr = load_and_prepare_image(test_image_path)
predicted_img = predict_sr(model, lr_img, (hr_img.shape[1], hr_img.shape[0]))

display_result(hr_img, predicted_img, original_lr)

print("LR Shape:", original_lr.shape)
print("Predicted SR Shape:", predicted_img.shape)
print("HR Shape:", hr_img.shape)
print("PSNR:", psnr(hr_img, predicted_img))