import sys
from PIL import Image as image
from PIL import ImageOps

emotions = ['happiness', 'sadness']
for i in range(1, 12):
    for emotion in emotions:
        img = image.open('C:/Users/jujum/OneDrive/Documents/IC/FacialRigImages/S'+str(i)+'/'+emotion+'.png')
        width, height = img.size
        box = (512, 512, 512, 512) # left, up, right, bottom
        crop = ImageOps.crop(img, box)
        crop.save('C:/Users/jujum/OneDrive/Documents/IC/FacialRigImages/S'+str(i)+'/'+emotion+'.png', 'PNG')