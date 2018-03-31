import numpy as np
import cv2
import sys
import os
import multiprocessing as mp


def rand_pair(max1, max2, size):
    for i in range(size):
        yield np.random.randint(max1), np.random.randint(max2)


def bgr888_to_bgr555(img):
    return img & 0xFC


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
    if np.any(np.var(data) > max_dist):
        slices = np.split(data, splits)
        total = []

        for s in slices:
            total += guess_colors(s, max_dist * 0.85, splits * 2)

        return total

    else:
        return [np.median(data, axis=0)]


def process_image(img):
    img = bgr888_to_bgr555(img)
    
    samp = sample(img, 256**2)
    samp = np.sort(samp, axis=0)

    colors = np.array(guess_colors(samp, 1500, 4))

    return colors


def main():
    path = sys.argv[1]

    in_data = [read_img(os.path.join(path, file)) for file in os.listdir(path)]
    pool = mp.Pool(processes=4)

    out_data = pool.map(process_image, in_data)
    max_len = max([len(x) for x in out_data])

    for i in range(len(out_data)):
        l = out_data[i].shape[0]
        out_data[i] = np.resize(out_data[i], (max_len, 3))
        out_data[i][l:, :] = 0

    cv2.imwrite('colors.png', np.array(out_data))


if __name__ == '__main__':
    main()
