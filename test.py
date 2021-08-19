from imutils import paths 
import cv2
from torchvision.transforms import transforms
from PIL import Image
import numpy as np

img_list = list(paths.list_images("../data/img_align_celeba"))

transform = transforms.Compose([
    transforms.Resize(178),
    transforms.CenterCrop(178),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

for path in img_list:
    img = Image.open(path).convert('RGB')
    
    img = transform(img)

    cv2.imshow("img", np.array(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()