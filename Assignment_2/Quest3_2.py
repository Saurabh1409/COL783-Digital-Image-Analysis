import numpy as np
import matplotlib.pyplot as plt
import cv2
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-white')

source_path = 'C:/Users/saura/Downloads/'

img= cv2.imread(source_path+'xdog.jpg',0)

plt.imshow(img,cmap='gray')
plt.show()

img = cv2.bilateralFilter(img,10,40,40)

img=np.asarray(img,dtype=np.double)

img  = cv2.resize(img,(200,200))

def lap(img):
    im = img.copy()
    g1 = cv2.GaussianBlur(im,(3,3),0)
    g2 = cv2.GaussianBlur(im,(5,5),0)
    dog = g1-g2
    smooth = cv2.GaussianBlur(im,(5,5),0)   
    down_samp = cv2.resize(smooth,(int(smooth.shape[0]/2),int(smooth.shape[1]/2)))
    return dog,down_samp

img1_gaussian_pyra = []
#img1_up_samp =[]
img1_lap_pyra = []

im =img.copy()

for i in range(6):
    l,d = lap(im)
    img1_gaussian_pyra.append(d)
    #img1_up_samp.append(u)
    img1_lap_pyra.append(l)
    im=d

fig,ax = plt.subplots(2,3,figsize=(14,12))
k=0
for i in range(2):
    for j in range(3):
        im=img1_gaussian_pyra[k]
        #ax[i].imshow(im[:,:,::-1].astype('uint8'))
        ax[i,j].imshow(im,cmap='gray')
        k+=1
ax[0,1].set_title("Gaussian Pyramid ")
plt.show()


fig,ax = plt.subplots(2,3,figsize=(14,12))
k=0
for i in range(2):
    for j in range(3):
        im=img1_lap_pyra[k]
        #ax[i].imshow(im[:,:,::-1].astype('uint8'))
        ax[i,j].imshow(im,cmap='gray')
        k+=1
ax[0,1].set_title("DOG Pyramid ")
plt.show()

f = img1_gaussian_pyra[-1]
f=cv2.resize(f,(img1_gaussian_pyra[-1].shape[0]*2,img1_gaussian_pyra[-1].shape[1]*2))

def thres(im):
    img=im.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j]  >= 60:
                img[i,j] = 1
            else:
                img[i,j] = 1+np.tanh(0.03*(img[i,j]-60))
                #S[i,j]=0
    return img

out = []
out_sketch = []
for i in range(5,-1,-1):
    l1 = 15*img1_lap_pyra[i]+f
    th1 = thres(l1)
    out.append(l1)
    out_sketch.append(th1)
    if(i==0):
        break
    #f = img1_gaussian_pyra[i-1]
    f=l1
    f=cv2.resize(f,(img1_lap_pyra[i-1].shape[0],img1_lap_pyra[i-1].shape[1]))

fig,ax = plt.subplots(2,3,figsize=(14,12))
k=0
for i in range(2):
    for j in range(3):
        im=out[k]
        #ax[i].imshow(im[:,:,::-1].astype('uint8'))
        ax[i,j].imshow(im,cmap='gray')
        k+=1
        
ax[0,1].set_title("Generated Image")
plt.show()

fig,ax = plt.subplots(2,3,figsize=(14,12))

k=0
for i in range(2):
    for j in range(3):
        im=out_sketch[k]
        #ax[i].imshow(im[:,:,::-1].astype('uint8'))
        ax[i,j].imshow(im,cmap='gray')
        k+=1
        
ax[0,1].set_title("xDOG Image")
plt.show()
