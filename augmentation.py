import torch
from torchvision.datasets           import ImageFolder
from torch                          import Tensor
from torchvision.transforms         import ToTensor
from torchvision.transforms         import transforms
from torchvision.transforms         import functional as TF
from torchvision.datasets.utils     import download_url
from PIL                            import Image
from PIL                            import ImageFilter
import os,os.path
import zipfile
from tqdm import tqdm
import shutil
import random 


def is_image(file_path):
    try:
        Image.open(file_path).verify()
        return True
    except (IOError, SyntaxError):
        return False
    
def download():
    dataset_url = "https://storage.googleapis.com/kaggle-data-sets/543939/992580/bundle/archive.zip"
    download_url(dataset_url, '.')  # Scarica il file nella directory corrente

def cleaningPath(data_directory):
    
    for fold in os.listdir(data_directory):
        path = os.path.join(data_directory, fold)
        if os.path.isdir(path):#unire tutto il path e fare il replace in quello
            for img in os.listdir(path):#
                if ' ' in img:
                    newimg_path = img.replace(' ','')
                    print(newimg_path)
                    os.rename(os.path.join(path, img), os.path.join(path,newimg_path) )           
                

def setname(dataset):
    for fold in os.listdir(dataset):
        path = os.path.join(dataset, fold)
        for img in os.listdir(path):
            #devo aggiungere al nome dell'immagine il nome della cartella
            newimg_path = fold + img
            os.rename(os.path.join(path, img), os.path.join(path,newimg_path) )
            
            
def centerCrop(dataset, output_size):
    new_dataset = []
    for img, label in tqdm(dataset, desc='Center Crop'):
        new_dataset.append((TF.center_crop(img, output_size), label))
    return new_dataset
        


def sharpen(dataset):
    new_dataset = []
    for img, label in tqdm(dataset, desc='Sharpen'):
        img = TF.adjust_sharpness(img, 2)
        new_dataset.append((img, label))
    return new_dataset


def resize(dataset, dim):
    new_dataset = []
    for img, label in tqdm(dataset, desc='Resizing Images'):
        new_dataset.append((TF.resize(img, dim, interpolation=Image.BICUBIC ,antialias=True), label))
    return new_dataset


def verticalFlip(dataset):
    new_dataset = []
    for img, label in tqdm(dataset, desc='Vertical Flip'):
        new_dataset.append((TF.vflip(img), label))
    return new_dataset

def horizontalFlip(dataset):
    new_dataset = []
    for img, label in tqdm(dataset, desc='Horizontal Flip'):
        new_dataset.append((TF.hflip(img), label))
    return new_dataset

def rotate(dataset):    
    new_dataset = []
    angle = random.randint(10,350)
    for img, label in tqdm(dataset, desc='Rotate'):
        new_dataset.append((TF.rotate(img, angle), label))
    return new_dataset

def grayScale(dataset): 
    new_dataset = []
    for img, label in tqdm(dataset, desc='Gray Scale'):
        img = TF.to_pil_image(img)
        new_dataset.append((TF.to_grayscale(img), label))
    return new_dataset

def brightness(dataset):
    new_dataset = []
    for img, label in tqdm(dataset, desc='Brightness'):
        new_dataset.append((TF.adjust_brightness(img, 1.5), label))
    return new_dataset

def contrast(dataset,val):
    new_dataset = []
    for img, label in tqdm(dataset, desc='Contrast'):
        new_dataset.append((TF.adjust_contrast(img, val), label))
    return new_dataset

def saturation(dataset,val):
    new_dataset = []
    for img, label in tqdm(dataset, desc='Saturation'):
        new_dataset.append((TF.adjust_saturation(img, val), label))
    return new_dataset

def hue(dataset,val):
    new_dataset = []
    for img, label in tqdm(dataset, desc='Hue'):
        new_dataset.append((TF.adjust_hue(img, val), label))
    return new_dataset

def gaussianBlur(dataset):
    new_dataset = []
    for img, label in tqdm(dataset, desc='Gaussian Blur'):
        new_dataset.append((TF.gaussian_blur(img, 3), label))
    return new_dataset

def colorJitter(dataset):
    new_dataset = []
    for img, label in tqdm(dataset, desc='Color Jitter'):
        color_jitter = transforms.ColorJitter(0.5, 0.5, 0.5, 0.5)
        new_dataset.append((color_jitter(img), label))
    return new_dataset


def randomVerticalShift(dataset):
    new_dataset = []
    for img, label in tqdm(dataset, desc='Random Vertical Shift'):
        new_dataset.append((TF.affine(img, angle=0, translate=(0, 0.2), scale =1.0, shear = 0.0), label))
    return new_dataset





def save(dataset, path, classes):
    if not os.path.exists(path):
        os.mkdir(path)
    for image, label in tqdm(dataset, desc='Save Images'):
        newpath = os.path.join(path, classes[label])
        if not os.path.exists(newpath):
            os.mkdir(newpath)
        if isinstance(image, Tensor):
            image = TF.to_pil_image(image)
        image.save(os.path.join(newpath, f"{str(random.randint(0, 100000))}.jpg"))



def main(DOWNLOAD):
    data_directory = 'C:/Users/lavoro/Desktop/universita/dataMining/lab/progetto/data'
    out_dir = 'C:/Users/lavoro/Desktop/universita/dataMining/lab/progetto/augmented'
    
    if(DOWNLOAD):
        download()
    
    #cleaningPath(out_dir)
    
    dataset = ImageFolder(data_directory, transform=ToTensor())
    classes = dataset.classes
    

    

    """dataset = resize(dataset, (256, 256))
    save(dataset, out_dir, classes)
    
    newdataset= centerCrop(dataset, 128)
    save(newdataset, out_dir, classes)

    newdataset = verticalFlip(dataset)
    save(newdataset, out_dir, classes)
    
    newdataset = sharpen(dataset)
    save(newdataset, out_dir, classes)

    newdataset = horizontalFlip(dataset)
    save(newdataset, out_dir, classes)
    
    newdataset = rotate(dataset)
    save(newdataset, out_dir, classes)
        
    newdataset = grayScale(dataset)
    save(newdataset, out_dir, classes)
    
    newdataset = brightness(dataset)
    save(newdataset, out_dir, classes)
    
    newdataset = contrast(dataset,1.5)
    save(newdataset, out_dir, classes)

    newdataset = contrast(dataset,0.8)
    save(newdataset, out_dir, classes)
    
    newdataset = saturation(dataset,1.5)
    save(newdataset, out_dir, classes)

    newdataset = saturation(dataset,0.5)
    save(newdataset, out_dir, classes)
    
    newdataset = hue(dataset,0.5)
    save(newdataset, out_dir, classes)

    newdataset = hue(dataset,-0.1)
    save(newdataset, out_dir, classes)

    newdataset = gaussianBlur(dataset)
    save(newdataset, out_dir, classes)
    
    newdataset = randomVerticalShift(dataset)
    save(newdataset, out_dir, classes)"""
    
    setname(out_dir)
    
    print("Augmentation completed")


if __name__ == '__main__':
    main(False) #true se il dataset deve ancora essere scaricato, false altrimenti