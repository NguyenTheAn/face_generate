import torch
from torchvision.transforms import transforms
import torch.utils.data as data
import torchvision
from imutils import paths
from PIL import Image

class Dataset(data.Dataset):
    
    def __init__(self, path):
        self.list_img = list(paths.list_images(path))

    def transform(self, image):
        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        image = transform(image)
        return image

    def __getitem__(self, index):
        img_pth = self.list_img[index]
        img = Image.open(img_pth).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.list_img)