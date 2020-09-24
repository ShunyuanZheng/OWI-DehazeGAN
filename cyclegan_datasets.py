"""Contains the standard train/test splits for the cyclegan data."""

"""The size of each dataset. Usually it is the maximum number of images from
each domain."""
DATASET_TO_SIZES = {
    'fog2defog_train': 2441,
    'fog2defog1_test': 45
}

"""The image types of each dataset. Currently only supports .jpg or .png"""
DATASET_TO_IMAGETYPE = {
    'fog2defog_train': '.jpg',
    'fog2defog1_test': '.jpg'
}

"""The path to the output csv file."""
PATH_TO_CSV = {
    'fog2defog_train': './input/fog2defog/fog2defog_train.csv',
    'fog2defog1_test': './input/test/fog2defog1_test.csv'
}

