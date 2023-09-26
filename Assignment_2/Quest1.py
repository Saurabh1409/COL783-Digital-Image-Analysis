
import dlib
import numpy as np
import cv2
from imutils import face_utils
import matplotlib.pyplot as plt
import imageio

source_path = 'C:/Users/saura/Downloads/'

srk = cv2.imread(source_path+'sor.jpg')
ran = cv2.imread(source_path+'randim.jpg')

fig,ax =plt.subplots(1,2)
ax[0].imshow(srk[:,:,::-1])
ax[1].imshow(ran[:,:,::-1])

srk = cv2.resize(srk,(200,200))
ran = cv2.resize(ran,(200,200))

fig,ax =plt.subplots(1,2)
ax[0].imshow(srk[:,:,::-1])
ax[1].imshow(ran[:,:,::-1])

def face_features(img1, img2):
    imgList = [img1,img2]
    l1 = []
    l2 = []
    j = 1
    
    # Detect the points of face.
    detec = dlib.get_frontal_face_detector()
    predic = dlib.shape_predictor(source_path+'shape_predictor_68_face_landmarks.dat')
    corresp = np.zeros((68,2))

    for img in imgList:

        size = (img.shape[0],img.shape[1])
        if(j == 1):
            Image = l1
        else:
            Image = l2

        dets = detec(img, 1)
        j=j+1

        for k, rect in enumerate(dets):
            
            # Get the landmarks/parts for the face in rect.
            shape = predic(img, rect)
            
            for i in range(0,68):
                x = shape.part(i).x
                y = shape.part(i).y
                Image.append([x, y])
                corresp[i][0] += x
                corresp[i][1] += y
                # cv2.circle(img, (x, y), 2, (0, 255, 0), 2)

            # Add back the background
            Image.append([1,1])
            Image.append([size[1]-1,1])
            Image.append([(size[1]-1)//2,1])
            Image.append([1,size[0]-1])
            Image.append([1,(size[0]-1)//2])
            Image.append([(size[1]-1)//2,size[0]-1])
            Image.append([size[1]-1,size[0]-1])
            Image.append([(size[1]-1),(size[0]-1)//2])

    # Add back the background
    narray = corresp/2
    narray = np.append(narray,[[1,1]],axis=0)
    narray = np.append(narray,[[size[1]-1,1]],axis=0)
    narray = np.append(narray,[[(size[1]-1)//2,1]],axis=0)
    narray = np.append(narray,[[1,size[0]-1]],axis=0)
    narray = np.append(narray,[[1,(size[0]-1)//2]],axis=0)
    narray = np.append(narray,[[(size[1]-1)//2,size[0]-1]],axis=0)
    narray = np.append(narray,[[size[1]-1,size[0]-1]],axis=0)
    narray = np.append(narray,[[(size[1]-1),(size[0]-1)//2]],axis=0)
    
    return [size,imgList[0],imgList[1],l1,l2,narray]

size,i1,i2,l1,l2,narray = face_features(srk,ran)

plt.imshow(i2[:,:,::-1])

fig,ax =plt.subplots(1,1,figsize=(10,8))
ax.imshow(i1[:,:,::-1])
ax.scatter(*zip(*l1),s = 12,c='r')


fig,ax =plt.subplots(1,1,figsize=(10,8))
ax.imshow(i2[:,:,::-1])
ax.scatter(*zip(*l2),s = 12,c='g')


def is_inside(rect, point):

    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True



def create_del(f_w, f_h, pointlist, img1, img2):

    # Make a rectangle.
    rect = (0, 0, f_w, f_h)

    # Create an instance of Subdiv2D.
    subdiv = cv2.Subdiv2D(rect)

    # Make a points list and a searchable dictionary. 
    pointlist = pointlist.tolist()
    points = [(int(x[0]),int(x[1])) for x in pointlist]
    dic = {x[0]:x[1] for x in list(zip(points, range(76)))}
    
    # Insert points into subdiv
    for p in points :
        subdiv.insert(p)

    # Make a delaunay triangulation list.

    # Write the delaunay triangles into a file
    
    trilist = subdiv.getTriangleList()
    list_del = []
    r = (0, 0, f_w, f_h)

    for t in trilist :
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        if is_inside(r, pt1) and is_inside(r, pt2) and is_inside(r, pt3) :
            list_del.append((dic[pt1],dic[pt2],dic[pt3]))
    # Return the list.
    return list_del

def tri_morph(img1, img2, img, t1, t2, t, alpha) :
    
    t1_r = []
    t2_r = []
    t_r = []

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))
    
    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    
    for i in range(0, 3):
        t_r.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1_r.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2_r.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    cv2.fillConvexPoly(mask, np.int32(t_r), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    im1_r = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    im2_r = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    #Affine Transform
    
    warp = cv2.getAffineTransform(np.float32(t1_r), np.float32(t_r))
    warp_im1 = cv2.warpAffine(im1_r, warp, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    
    warp = cv2.getAffineTransform(np.float32(t2_r), np.float32(t_r))
    warp_im2 = cv2.warpAffine(im2_r, warp, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # Alpha blend rectangular patches
    img_rect = (1.0 - alpha) * warp_im1 + alpha * warp_im2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + img_rect * mask

def morph_out(img1,img2,points1,points2,tri_list,size):

    num_images=20
    out=[]
    del_line_img = []
    alpha=0
    for j in range(0, num_images):

        # Convert Mat to float data type
        img1 = np.float32(img1)
        img2 = np.float32(img2)

        # Read array of corresponding points
        points = []
        alpha += (1/num_images)

        # Compute weighted average point coordinates
        for i in range(0, len(points1)):
            x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
            y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
            points.append((x,y))
        
        # Allocate space for final output
        morphed_frame = np.zeros(img1.shape, dtype = img1.dtype)
        m = morphed_frame.copy()

        for i in range(len(tri_list)):    
            x = int(tri_list[i][0])
            y = int(tri_list[i][1])
            z = int(tri_list[i][2])
            
            t1 = [points1[x], points1[y], points1[z]]
            t2 = [points2[x], points2[y], points2[z]]
            t = [points[x], points[y], points[z]]

            # Morph one triangle at a time.
            tri_morph(img1, img2, morphed_frame, t1, t2, t, alpha)
            tri_morph(img1, img2, m, t1, t2, t, alpha)
            
            pt1 = (int(t[0][0]), int(t[0][1]))
            pt2 = (int(t[1][0]), int(t[1][1]))
            pt3 = (int(t[2][0]), int(t[2][1]))
            

            
            
            cv2.line(m, pt1, pt2, (0, 255, 0),1,8, 0)
            cv2.line(m, pt2, pt3, (0, 255, 0),1,8, 0)
            cv2.line(m, pt3, pt1, (0, 255, 0),1,8, 0)

        tex =cv2.cvtColor(np.uint8(m), cv2.COLOR_BGR2RGB)
        del_line_img.append(tex)
            
        temp = cv2.cvtColor(np.uint8(morphed_frame), cv2.COLOR_BGR2RGB)
        out.append(temp.astype('uint8'))

    return out,del_line_img


img1=srk
img2=ran


[size, img1, img2, points1, points2, list3] = face_features(img1, img2)


tri = create_del(size[1], size[0], list3, img1, img2)

out,del_line=morph_out(img1, img2, points1, points2, tri, size)


imageio.mimsave(source_path+'Q1.gif',out,fps=6)



fig,ax =plt.subplots(1,1,figsize=(10,8))
ax.imshow(out[12])


fig,ax =plt.subplots(1,2,figsize=(15,12))
ax[0].imshow(del_line[0])
ax[1].imshow(del_line[-1])
plt.show()




