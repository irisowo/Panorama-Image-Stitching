import exifread
import os
import csv
import cv2
import sys


def read_focal_length(img_name):
    with open(img_name, 'rb') as f:
        tags = exifread.process_file(f)
        focal_length = tags.get('EXIF FocalLength')
        if focal_length:
            frac_f = focal_length.values[0].num / focal_length.values[0].den
            frac_f *= 120
            return int(frac_f)
        else:
            return int(800)
            # raise ValueError("Focal length is not found in the image metadata.")


def create_focal_length_csv(img_dir):
    csv_name = 'focal_length.csv'
    csv_file = os.path.join(img_dir, csv_name)
    columns = ['filename', 'focal_length']
    def is_image_file(file_name):
        ext = file_name.split('.')[-1]
        return ext.lower() in ['jpg', 'jpeg', 'png', 'bmp', 'gif']

    img_names = [f for f in os.listdir(img_dir) if is_image_file(f)]

    with open(csv_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for img_name in sorted(img_names):
            focal_length = read_focal_length(os.path.join(img_dir, img_name))
            writer.writerow([img_name, focal_length])


def resize_image(img_dir, resized_img_dir, scale=0.3):
    if not os.path.exists(resized_img_dir):
        os.makedirs(resized_img_dir)
    for img_name in os.listdir(img_dir):
        if not img_name.endswith('.JPG'):
            continue
        img = cv2.imread(os.path.join(img_dir, img_name))
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        cv2.imwrite(os.path.join(resized_img_dir, img_name), img)


if __name__ == "__main__":
    # Testing
    # img_name = "../data/home/IMG_4242.JPG"
    # focal_length = read_focal_length(img_name)
    # print(f"Focal length: {focal_length}")

    img_dir = sys.argv[1]
    print(f'Processing images in {img_dir}')
    dataset_name = os.path.basename(img_dir)
    img_out_dir = os.path.join(img_dir, f'../{dataset_name}_resized')
    resize_image(img_dir, img_out_dir, 1.0)
    create_focal_length_csv(img_out_dir)
