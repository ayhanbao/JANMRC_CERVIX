import os

import cv2

from tools.img_utils import norm_path, split_path, image_list

IMAGE_SIZE = 512


def resize_image(im, height=IMAGE_SIZE, width=IMAGE_SIZE):

    def get_padding_size(im):
        h, w, _ = im.shape
        longest_edge = max(h, w)
        top, bottom, left, right = (0, 0, 0, 0)
        if h < longest_edge:
            dh = longest_edge - h
            top = dh // 2
            bottom = dh - top
        elif w < longest_edge:
            dw = longest_edge - w
            left = dw // 2
            right = dw - left
        else:
            pass
        return top, bottom, left, right

    top, bottom, left, right = get_padding_size(im)
    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    # resize image
    resized_image = cv2.resize(constant, (height, width))
    return resized_image


def resize_image_file(img_path, image_size, save_path):
    img_path = (norm_path(img_path))
    save_path = norm_path(save_path)

    save_dir, _, _ = split_path(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # read image
    im = cv2.imread(img_path)

    # resize image
    im = resize_image(im, image_size)

    # write image
    cv2.imwrite(save_path[:-3] + 'png', im)


def main(src_path, dst_path, image_size):
    try:
        for file in image_list(src_path):
            _, name, ext = split_path(file)
            save = os.path.join(dst_path, name + ext)
            print(save)
            resize_image_file(file, image_size, save)
    except IOError:
        pass  # You can always log it to logger


if __name__ == '__main__':
    src_path = '/home/bong6/data/mrcnn_cer/stage1_train/mask'
    dst_path = '/home/bong6/data/mrcnn_cer/stage1_train/mask1'

    main(src_path, dst_path, IMAGE_SIZE)
