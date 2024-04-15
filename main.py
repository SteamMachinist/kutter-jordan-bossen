import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from bitarray import bitarray


def string_to_bits_np_array(string):
    bit_array = bitarray()
    bit_array.frombytes(string.encode('ascii'))
    return np.array(bit_array.tolist())


def bits_np_array_to_string(bit_np_array):
    bit_array = bitarray()
    bit_array.extend(bit_np_array)
    return bit_array.tobytes().decode('ascii', errors='replace')


def add_postfix(filename):
    name, extension = os.path.splitext(filename)
    return f"{name}_encoded{extension}"


def pixel_brightness(image, x, y): #BGR
    return 0.299 * image[y, x, 2] + 0.587 * image[y, x, 1] + 0.144 * image[y, x, 0]


def encode_message_to_image(image_path, message, key, sigma, repeat_times, q=2):
    message_bits = string_to_bits_np_array(message) #[0, 1]
    multipliers = np.where(message_bits == 0, -1, message_bits) #[-1, 1]
    multipliers = np.repeat(multipliers, repeat_times) #[-1, -1, -1, -1, -1, 1, 1, 1, 1, 1]

    image = cv2.imread(image_path)
    height, width, _ = image.shape

    np.random.seed(key)
    for bit in multipliers:
        x = np.random.randint(sigma, width - sigma)
        y = np.random.randint(sigma, height - sigma)

        l = pixel_brightness(image, x, y)
        image[y, x, 0] = np.clip(int(image[y, x, 0] + bit * q * l), 0, 255)

    cv2.imwrite(add_postfix(image_path), image)


def decode_message_from_image(image_path, key, sigma, repeat_times, meaningful_bits):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    bits_with_repeats = []
    np.random.seed(key)
    for _ in range(sigma, width - sigma):
        for _ in range(sigma, height - sigma):
            x = np.random.randint(sigma, width - sigma)
            y = np.random.randint(sigma, height - sigma)

            actual = image[y, x, 0]

            blue_sum = 0
            for i in range(1, sigma + 1):
                blue_sum = (blue_sum +
                            image[y + i, x, 0] +
                            image[y - i, x, 0] +
                            image[y, x + i, 0] +
                            image[y, x - i, 0])
            predicted = blue_sum / (4 * sigma)

            bits_with_repeats.append(1 if actual > predicted else 0)

    bits = []
    for i in range(0, len(bits_with_repeats) - repeat_times, repeat_times):
        bits_of_one = bits_with_repeats[i: i + repeat_times]
        bits.append(round(sum(bits_of_one) / len(bits_of_one)))

    return bits_np_array_to_string(np.array(bits)[:meaningful_bits])


def show_images(original_image_name):
    original_image_rgb = cv2.cvtColor(cv2.imread(original_image_name), cv2.COLOR_BGR2RGB)

    encoded_image_name = add_postfix(original_image_name)
    encoded_image_rgb = cv2.cvtColor(cv2.imread(encoded_image_name), cv2.COLOR_BGR2RGB)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7))

    ax1.imshow(original_image_rgb)
    ax1.set_title('Original image')
    ax1.axis('off')

    ax2.imshow(encoded_image_rgb)
    ax2.set_title('Encoded image')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()


message = "Hello, World!"
image_name = 'Volume_91.jpg'

key = 4242
sigma = 3
repeat_times = 5

print("Original message: " + message)

encode_message_to_image(image_name, message, key, sigma, repeat_times)

print("Decoded message:  " +
      decode_message_from_image(add_postfix(image_name), key, sigma, repeat_times, 200))

show_images(image_name)
