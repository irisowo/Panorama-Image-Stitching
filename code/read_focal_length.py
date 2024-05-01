import exifread
import os
import csv
import cv2
import sys


def is_image_file(file_name):
    ext = file_name.split('.')[-1]
    return ext.lower() in ['jpg', 'jpeg', 'png', 'bmp', 'gif']


def read_focal_length_from_img(img_name):
    with open(img_name, 'rb') as f:
        tags = exifread.process_file(f)
        focal_length = tags.get('EXIF FocalLength')
        if focal_length:
            frac_f = focal_length.values[0].num / focal_length.values[0].den
            frac_f *= 120
            print(f'focal = {frac_f}')
            return int(frac_f)
        else:
            return int(1000)
            # raise ValueError("Focal length is not found in the image metadata.")


def read_focal_length_from_txt(txt_name):
    with open(txt_name, 'r') as f:
        lines = f.readlines()
        focal_lengths = []
        for i in range(11, len(lines), 13):
            focal_length = float(lines[i].split('\n')[0])
            focal_lengths.append(focal_length)
    return focal_lengths


def create_focal_length_csv(img_dir, img_out_dir):

    csv_name = 'focal_length.csv'
    columns = ['filename', 'focal_length']
    img_names = sorted([f for f in os.listdir(img_dir) if is_image_file(f)])

    with open(os.path.join(img_out_dir, csv_name), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        # if pano.txt exists, read from pano.txt

        if os.path.exists(os.path.join(img_dir, 'pano.txt')):
            focal_lengths = read_focal_length_from_txt(os.path.join(img_dir, 'pano.txt'))
            for img_name, focal_length in zip(img_names, focal_lengths):
                writer.writerow([img_name, focal_length])
        else:
            for img_name in img_names:
                focal_length = read_focal_length_from_img(os.path.join(img_dir, img_name))
                writer.writerow([img_name, focal_length])


def resize_image(img_dir, resized_img_dir, scale=0.3):
    if not os.path.exists(resized_img_dir):
        os.makedirs(resized_img_dir)
    for img_name in os.listdir(img_dir):
        if not is_image_file(img_name):
            continue
        img = cv2.imread(os.path.join(img_dir, img_name))
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        cv2.imwrite(os.path.join(resized_img_dir, img_name), img)


if __name__ == "__main__":
    img_dir = sys.argv[1]
    print(f'Processing images in {img_dir}')
    dataset_name = os.path.basename(img_dir)

    img_out_dir = os.path.join(img_dir, f'../{dataset_name}_resized')
    resize_image(img_dir, img_out_dir, 0.5)
    create_focal_length_csv(img_dir, img_out_dir)
