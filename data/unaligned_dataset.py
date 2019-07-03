import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import cv2

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.movie_path = opt.movie
        #os.path.join(opt.dataroot, opt.movie)

        input_nc = self.opt.input_nc       # get the number of channels of input image

        self.cap = cv2.VideoCapture(opt.movie)
        frames_num = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) #clay changed cap to self.cap



        self.A_size = frames_num

        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """

        if self.opt.serial_batches:   # make sure index is within then range
            index_A = index % self.A_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_A = random.randint(0, self.A_size - 1)

        #while (cap.isOpened()):
        ret, frame = self.cap.read() #clay cap to self.cap

        #if frame == None:
        #    return None

        #A_img = Image.open(A_path).convert('RGB')
        A_img = Image.fromarray(frame).convert('RGB')
        # apply image transformation
        A = self.transform_A(A_img)

        return A

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.A_size # clay removed max(self.A_size, self.B_size)
