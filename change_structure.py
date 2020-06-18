"""
change the data structure
"""



import argparse
import random
import os

from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/data_new', help="Directory with the old data dataset")
parser.add_argument('--output_dir' , default='data/data_new_2', help="Where to write the new new_data_2")


if __name__ == '__main__':

	args = parser.parse_args()
	assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)
	
