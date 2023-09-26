
import cv2
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-white')
import imageio
import warnings
warnings.filterwarnings("ignore")

def face_detec(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(gray, (x,y), (x+w, y+h), (0, 255, 0), 2)
        img_crop=img[y:y+h, x:x+w]
        
    return img_crop

source_path = 'C:/Users/saura/Downloads/'

img1=cv2.imread(source_path+'randim.jpg')
img2=cv2.imread(source_path+'sor.jpg')

face1 = face_detec(img1)

plt.imshow(face1[:,:,::-1])
plt.show()

face2 = face_detec(img2)

plt.imshow(face2[:,:,::-1])
plt.show()

img1 =cv2.resize(face1,(200,200))
img2 =cv2.resize(face2,(200,200))


img1=np.asarray(img1,dtype=np.double)
img2=np.asarray(img2,dtype=np.double)

def lap(img):
    im = img.copy()
    smooth = cv2.GaussianBlur(im,(5,5),0)
    down_samp = cv2.resize(smooth,(int(smooth.shape[0]/2),int(smooth.shape[1]/2)))
    up_samp = cv2.resize(down_samp,(int(down_samp.shape[0]*2),int(down_samp.shape[1]*2)))
    up_samp = cv2.GaussianBlur(up_samp,(5,5),0)
    if(im.shape[0]!=up_samp.shape[0] or im.shape[1]!=up_samp.shape[1]):
        up_samp = cv2.resize(up_samp,(im.shape[0],im.shape[1]))
    lap_im = im - up_samp
    return lap_im,down_samp,up_samp

img1_gaussian_pyra = []
img1_up_samp =[]
img1_lap_pyra = []

img =img1.copy()

for i in range(4):
    l,d,u = lap(img)
    img1_gaussian_pyra.append(d)
    img1_up_samp.append(u)
    img1_lap_pyra.append(l)
    img=d

fig,ax = plt.subplots(1,4)
for i in range(4):
    im=img1_gaussian_pyra[i]
    ax[i].imshow(im[:,:,::-1].astype('uint8'))
    #ax[i].imshow(im,cmap='gray')
plt.show()

fig,ax = plt.subplots(1,4)
for i in range(4):
    im=img1_lap_pyra[i]
    ax[i].imshow(im[:,:,::-1].astype('uint8'))
    #ax[i].imshow(im,cmap='gray')
plt.show()

img2_gaussian_pyra = []
img2_up_samp =[]
img2_lap_pyra = []

img =img2.copy()

for i in range(4):
    l,d,u = lap(img)
    img2_gaussian_pyra.append(d)
    img2_up_samp.append(u)
    img2_lap_pyra.append(l)
    img=d

fig,ax = plt.subplots(1,4)
for i in range(4):
    im=img2_gaussian_pyra[i]
    ax[i].imshow(im[:,:,::-1].astype('uint8'))
    #ax[i].imshow(im,cmap='gray')
plt.show()

fig,ax = plt.subplots(1,4)
for i in range(4):
    im=img2_lap_pyra[i].copy()
    ax[i].imshow(im[:,:,::-1].astype('uint8'))
    #ax[i].imshow(im,cmap='gray')
plt.show()

mask = np.zeros((img1.shape[0], img1.shape[1]))

out=[]
for i in range(32):
    mask = np.zeros((img1.shape[0], img1.shape[1]))
    mask[:,:int(img1.shape[1]/32)*i]=1
    t = mask
    mask_pyra = [mask]
    for i in range(5):
        b = cv2.GaussianBlur(t,(5,5),0)
        d= cv2.resize(b,(int(b.shape[0]/2),int(b.shape[1]/2)))
        mask_pyra.append(d)
        t=d
    m1 = mask_pyra[4]
    m2 = 1-mask_pyra[4]

    m1 = np.stack(np.array([m1,m1,m1]), axis=2)
    m2 = np.stack(np.array([m2,m2,m2]), axis=2)
    g = (img1_gaussian_pyra[3]*m1) + (img2_gaussian_pyra[3]*m2)
    g1=cv2.resize(g,(int(img1_gaussian_pyra[2].shape[0]),int(img1_gaussian_pyra[2].shape[1])))
    g1 = cv2.GaussianBlur(g1,(5,5),0)

    final = []
    for i in range(3,-1,-1):

        m1 = mask_pyra[i]
        m2 = 1-mask_pyra[i]
        m1 = np.stack(np.array([m1,m1,m1]), axis=2)
        m2 = np.stack(np.array([m2,m2,m2]), axis=2)

        #print(g1.shape)
        l1= (img1_lap_pyra[i]*m1) + (img2_lap_pyra[i]*m2)

        blend = l1  + g1
        final.append(blend)

        g1= cv2.resize(blend,(0,0),fx=2,fy=2)
        g1 = cv2.GaussianBlur(g1,(5,5),0)
        
    out.append(final[-1])

out_gif = []
for i in range(32):
    im1=out[i]
    im = np.zeros((im1.shape[0],im1.shape[1],im1.shape[2]))
    for k in range(im.shape[0]):
        for j in range(im.shape[1]):
            for l in range(3):
                if im1[k,j,l]>255:
                    im[k,j,l]=255
                elif im1[k,j,l]<0:
                    im[k,j,l]=0
                else:
                    im[k,j,l]=im1[k,j,l]
    im = np.asarray(im,dtype=np.uint8)
    #print(im)
    out_gif.append(im[:,:,::-1])
    #ax[i].imshow(im[:,:,::-1])

imageio.mimsave(source_path+'Q2.gif',out_gif,fps=6)


