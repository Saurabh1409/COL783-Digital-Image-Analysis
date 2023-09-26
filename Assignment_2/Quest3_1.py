import numpy as np
import matplotlib.pyplot as plt
import cv2

source_path = 'C:/Users/saura/Downloads/'

img= cv2.imread(source_path+'xdog.jpg',0)

plt.imshow(img,cmap='gray')
plt.show()

img = cv2.bilateralFilter(img,10,40,40)
img=np.asarray(img,dtype=np.double)

g1= cv2.GaussianBlur(img,(5,5),0.8)
plt.imshow(g1,cmap='gray')
plt.show()

g1k= cv2.GaussianBlur(img,(5,5),1.28)
plt.imshow(g1k,cmap='gray')

dog = g1-g1k
dog = 60*dog

plt.imshow(dog,cmap='gray')
plt.show()

S=g1+dog

plt.imshow(S,cmap='gray')
plt.show()

for i in range(S.shape[0]):
    for j in range(S.shape[1]):
        if S[i,j]  >= 80:
            S[i,j] = 1
        else:
            S[i,j] = 1+np.tanh(0.04*(S[i,j]-80))
            #S[i,j]=0

fig,ax = plt.subplots(1,2,figsize=(14,12))
ax[0].imshow(img,cmap='gray')
ax[1].imshow(S,cmap='gray')
plt.show()


