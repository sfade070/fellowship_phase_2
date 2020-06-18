"""
Split the SIGNS dataset into train/val/test and resize images to 64x64.

The SIGNS dataset comes into the following format:
    train_signs/
        0_IMG_5864.jpg
        ...
    test_signs/
        0_IMG_5942.jpg
        ...

Original images have size (3024, 3024).
Resizing to (64, 64) reduces the dataset size from 1.16 GB to 4.7 MB, and loading smaller images
makes training faster.

We already have a test set created, so we only need to split "train_signs" into train and val sets.
Because we don't have a lot of images and we want that the statistics on the val set be as
representative as possible, we'll take 20% of "train_signs" as val set.
"""

import argparse
import random
import os

from PIL import Image
from tqdm import tqdm
import shutil
SIZE = 64*4

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/data_new', help="Directory with the data_new dataset")
parser.add_argument('--output_dir' , default='data/data_new_2', help="Where to write the new data")





def resize_and_save(filename, output_dir, size=SIZE):
    """ Resize the image contained in `filename` and save it to the `output_dir` """
    image = Image.open(filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((size, size), Image.BILINEAR)
    image.save(os.path.join(output_dir, filename.split('/')[-1]))





if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    if not os.path.exists(os.path.join(args.data_dir, "train_0")):
    	os.makedirs(os.path.join(args.data_dir, "train_0"))

    if not os.path.exists(os.path.join(args.data_dir, "test_0")):
    	os.makedirs(os.path.join(args.data_dir, "test_0"))	


    train_data_dir =  os.path.join(args.data_dir, 'train_0')
    test_data_dir  =  os.path.join(args.data_dir, 'test_0')


    # Define the data directories
    for d in os.listdir(args.data_dir):
        if d not in [train_data_dir.split('/')[-1],test_data_dir.split('/')[-1]]:
            l = os.listdir(os.path.join(args.data_dir, d))
            random.shuffle(l)
            for f in l:
                u = random.uniform(0,1)
#                if (not f.endswith('.jpg')):
#                    print(f)
                if (u < 0.1):
                    shutil.move(os.path.join(args.data_dir, d,f), test_data_dir)
                else:
                    shutil.move(os.path.join(args.data_dir, d,f), train_data_dir)





    # Get the filenames in each directory (train and test)
    filenames = os.listdir(train_data_dir)
    filenames = [os.path.join(train_data_dir, f) for f in filenames if (f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png'))]

    test_filenames = os.listdir(test_data_dir)
    test_filenames = [os.path.join(test_data_dir, f) for f in test_filenames if (f.endswith('.jpg') or f.endswith('.jpeg')or f.endswith('.png'))]

    # Split the images in 'train_signs' into 80% train and 20% val
    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    random.seed(230)
    filenames.sort()
    random.shuffle(filenames)

    split = int(0.8 * len(filenames))
    train_filenames = filenames[:split]
    val_filenames = filenames[split:]

    filenames = {'train': train_filenames,
                 'val': val_filenames,
                 'test': test_filenames}

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    # Preprocess train, val and test
    for split in ['train', 'val', 'test']:
        output_dir_split = os.path.join(args.output_dir, '{}_signs'.format(split))
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print("Warning: dir {} already exists".format(output_dir_split))

        print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
        for filename in tqdm(filenames[split]):
            resize_and_save(filename, output_dir_split, size=SIZE)

    print("Done building dataset")
