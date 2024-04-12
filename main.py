import random

from PIL import Image, ImageDraw, PyAccess
from bitarray._bitarray import bitarray
from bitarray.util import int2ba

text = bytes('Hello, world!', 'cp866')
L = 0.5
sigma = 4
K0 = 123

image = Image.open("1684384546_1234_w2xex_2x_jpg.jpg")
draw = ImageDraw.Draw(image)
width, height = image.size
pix:PyAccess = image.load()

bit_chars = []
for char in text:
    bites = [1 if i else 0 for i in int2ba(char)]
    bit_chars.append(([0] * (8 - len(bites)) + bites))

random.seed(K0)
print(bit_chars)
for i in range(len(bit_chars)):
    for j in range(len(bit_chars[i])):
        x = random.uniform(sigma, width - sigma)
        y = random.uniform(sigma, height - sigma)

        r = pix[x, y][0]
        g = pix[x, y][1]
        b = pix[x, y][2]
        power = 0.299 * r + 0.589 * g + 0.114 * b

        b += int((2 * bit_chars[i][j] - 1) * L * power)

        if b < 0:
            b = 0
        if b > 255:
            b = 255
        draw.point((x, y), (r, g, b))


image.save("encrypted.png", "PNG")

image = Image.open("encrypted.png")
pix = image.load()
random.seed(K0)

width = image.size[0]
height = image.size[1]
res = bitarray()
for i in range(sigma, width - sigma):
    for j in range(sigma, height - sigma):
        x = random.uniform(sigma, width - sigma)
        y = random.uniform(sigma, height - sigma)

        temp_b = 0
        b = pix[x, y][2]
        for k in range(1, sigma + 1):
            bt = pix[x, y + k][2]
            bd = pix[x, y - k][2]
            bl = pix[x - k, y][2]
            br = pix[x, y + k][2]
            temp_b += bt + bd + bl + br
        temp_b /= 4 * sigma

        res += [temp_b < b]

res = res[:len(text) * 8]
print('Результат расшифровывания: ', res.tobytes().decode('cp866'))
ba = bitarray(endian='big')
ba.frombytes(text)
print('Ошибки расшифровывания: ', res.__xor__(ba).count(1))
