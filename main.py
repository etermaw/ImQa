import numpy as np
import cv2
import sys
import os


def rand_pair(max1, max2, size):
    for i in range(size):
        yield np.random.randint(max1), np.random.randint(max2)


def bgr888_to_bgr555(img):
    return img & 0xFC


def fuse_bgr(img):
    return (img[:, :, 2] << 16) | (img[:, :, 1] << 8) | img[:, :, 0]


def read_img(name):
    return cv2.imread(name)


def sample(img, count):
    i = 0
    ret = np.zeros((count, 3))

    for x, y in rand_pair(img.shape[0], img.shape[1], count):
        ret[i] = img[x, y]
        i += 1

    return ret


def guess_colors(data, max_dist, splits):
    med = np.median(data, axis=0)
    avg = np.mean(data, axis=0)

    if np.any(np.abs(med - avg) > max_dist):
        slices = np.split(data, splits)
        total = []

        for s in slices:
            total += guess_colors(s, max_dist, splits)

        return total

    else:
        return [med]


def fuse_colors(data, dist):
    ret = []
    prev = data[0]

    for pixel in data:
        if np.all(np.abs(pixel - prev) <= dist):
            ret.append(pixel)

        prev = pixel

    return np.array(ret)


def main():
    dir = sys.argv[1]
    c = 0
    s = 256

    guessed_colors = []

    for file in os.listdir(sys.argv[1]):
        img = read_img(os.path.join(dir, file))
        img = bgr888_to_bgr555(img)

        rand = sample(img, s**2)
        rand = np.sort(rand, axis=0)

        ret = np.array(guess_colors(rand, 5, 4))
        guessed_colors.append(ret)

        print('{} -> out{}.png'.format(file, c))

        rand = np.reshape(rand, (s, s, 3))
        cv2.imwrite('out{}.png'.format(c), rand)
        c += 1

    max_len = max([len(x) for x in guessed_colors])

    for i in range(len(guessed_colors)):
        l = guessed_colors[i].shape[0]
        guessed_colors[i] = np.resize(guessed_colors[i], (max_len, 3))
        guessed_colors[i][l:, :] = 0

    cv2.imwrite('colors.png', np.array(guessed_colors))


if __name__ == '__main__':
    main()
