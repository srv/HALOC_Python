####################################################################################################################################################################################
# Author	: Francisco Bonin Font
# History     : 27-June-2019 - Creation
# NOTES:
#	1. This code runs fine with python 3.7.3. inside an Anaconda environment (https://www.anaconda.com/distribution/?gclid=EAIaIQobChMImM39x9Ow5gIVhYxRCh3CXgvnEAAYASAAEgI91vD_BwE). 
# 	2. Not tested on other versions of python.
# 	3. You will need to install OpenCv for python: pip install opencv-python 
# 	4. Take into account that , if you are using Jupyter Notebook, it is necessary to run, first: import sys
# 	5.if you have installed ROS (Robot Operating System) Kinetic or any other ROS distribution, first you will need to deactivate the Python lybraries installed with ROS:  
# 			sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
#####################################################################################################################################################################################


import matplotlib.pyplot as plt
import cv2 # import open CV for image processing: SIFT features
import numpy as np # mathematic operations
import os # utility for access to directories. 




def imshow(openCVImage):
    plt.figure()
    plt.imshow(cv2.cvtColor(openCVImage,cv2.COLOR_BGR2RGB))
    

def get_descriptors(theImage,num_max_fea):

    gsImage=cv2.imread(theImage,cv2.IMREAD_GRAYSCALE) # read the image "theImage" from the hard disc in gray scale and loads it as a OpenCV CvMat structure. 
  
   	# gsImage=cv2.cvtColor(curImage,cv2.COLOR_BGR2GRAY) # convert image to gray scale
	#plt.figure()
	#plt.imshow(gsImage) # shows image
    theSIFT=cv2.xfeatures2d.SIFT_create((num_max_fea-3)) # creates a object type SIFT 
    keyPoints,theDescriptors=theSIFT.detectAndCompute(gsImage,None) # keypoint detection and descriptors descriptors, sense mascara
   # img=cv2.drawKeypoints(gsImage,keyPoints,curImage,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # uncomment in case you want to see the keypoints
   # plt.figure()
   # plt.imshow(img)
    nbr_of_keypoints=len(keyPoints)
    # sanity checks: 
    if nbr_of_keypoints==0:
        print("ERROR: descriptor Matrix is Empty")
        return 
    if nbr_of_keypoints>len(vector1):
        print("ERROR:  The number of descriptors is larger than the size of the projection vector. This should not happen.")
        return

    num_of_descriptors=theDescriptors.shape[0] #--> 100
    num_of_components=theDescriptors.shape[1] # --> 128
    hash=[]
    dot = 0
    dot_normalized=0
    suma = 0

    for i in range(num_of_components):
        suma=0
        for j in range(num_of_descriptors):
            dot = theDescriptors[j,i]*vector1[j] # for a fixed component, (a fixed column) vary the descriptor (row): dot product 
#between the matrix column and the vector
            dot_normalized = (dot + 1.0) / 2.0
            suma = suma + dot_normalized
        
        hash=np.append(hash, (suma/num_of_descriptors))   

    for i in range(num_of_components):
        suma=0
        for j in range(num_of_descriptors):
            dot = theDescriptors[j,i]*vector2[j] # for a fixed component, (a fixed column) vary the descriptor (row): dot product 
#between the matrix column and the vector
            dot_normalized = (dot + 1.0) / 2.0
            suma = suma + dot_normalized
        
        hash=np.append(hash, (suma/num_of_descriptors))   


    for i in range(num_of_components):
        suma=0
        for j in range(num_of_descriptors):
            dot = theDescriptors[j,i]*vector3[j] # for a fixed component, (a fixed column) vary the descriptor (row): dot product 
#between the matrix column and the vector
            dot_normalized = (dot + 1.0) / 2.0
            suma = suma + dot_normalized
        
        hash=np.append(hash, (suma/num_of_descriptors))   
    return hash


#############################################################3
qPath='/home/xesc/RECERCA/SUMMUM/CNN/netvlad-master/img_datasets/valldemossa/queries_75_566/' # write here the global path of your queries
dBPath='/home/xesc/RECERCA/SUMMUM/CNN/netvlad-master/img_datasets/valldemossa/Dbase_75_566/' # write here the global path of the corresponding database of images
num_max_features=100 # define the maximum number of features

 # get the 3 orthogonal unitary vectors

vector1=np.random.uniform(0,1,num_max_features); # creates a vector of random numbers between 0 and 1
vector1 /= np.linalg.norm(vector1) # normalize vector 1
# now create two vectors orthogonal to vector1 and with module = 1
vector2 = np.random.uniform(0,1,(num_max_features-1)); # second random vector, one less  component 
const1=0
long=num_max_features-1
for i in range(long): # dot product between vector2 and vector 1 for the num_max_features-1 components
    const1=const1+(vector1[i]*vector2[i])

xn=-const1/vector1[num_max_features-1] # the last component of vector2 and the one that makes vector1·vector2=0
vector2=np.append(vector2, xn) # add the last component to vector2. Now, vector 1 and vector 2 are orthogonals

vector2 /= np.linalg.norm(vector2) # normalitzo vector2 otra vez
vector3 = np.random.uniform(0,1,(num_max_features-2)); # create another vector , random and unitary


# vector 3 is orthogonal to vector1 and vector2, forcing all components to be random except the two last, which will result from solving a system of two equations with two variables and 
# where the scalar product of vector 3 with vector1 and vector2 must be 0 


const1=0
const2=0
long=num_max_features-2
for i in range(long): # dot product between vector3 and the num_max_features-2 components of vector1 and vector2
    const1=const1+(vector1[i]*vector3[i])
    const2=const2+(vector2[i]*vector3[i])

# force the last two elements of vector3 to be orthogonal to vector1 and vector2. Solve a linear system of 
# equations Ax=B, where A --> the last two components of vector1 and vector2, in the form of
# two rows of A, row 1 = vector1, row 2 = vector2. B is the constant components, taken from the 
# dot product between the first num_max_features-2 components of vector1 and the num_max_features-2 components of vector2, 
# with all the random components of vector3. And X are the last two components of vector3, in such a way that
# vector1 · vector3=0 and vector2 · vector3=0. 
A = np.array([[vector1[num_max_features-2],vector1[num_max_features-1]], [vector2[num_max_features-2],vector2[num_max_features-1]]])
B = np.array([-const1,-const2])
X = np.linalg.solve(A, B) # solve the linear system. X[0] is the penultimate element of vector3 ,X[1] is the last element of vector3
#np.allclose(np.dot(A, X), B) # true if Ax=B

vector3=np.append(vector3, X[0]) ## append the last two elements to vector3
vector3=np.append(vector3, X[1])
vector3 /= np.linalg.norm(vector3) # normalitzo vector3

  #  print ("lengh of vector3: "+str(len(vector3)))

print(np.linalg.norm(vector1), np.linalg.norm(vector2), np.linalg.norm(vector3)) # just visualize some results
print(np.dot(vector1, vector2) , np.dot(vector3, vector2) , np.dot(vector1, vector3))

#select one query
query_image=os.path.join(qPath,'153_85.jpg') # the global  path of the query image
hash_query=get_descriptors(query_image,num_max_features) # get the query hash

allFiles=[x for x in os.listdir(dBPath) if x.upper().endswith('.JPG')] ## list of images of the db directory. Assume that all are JPG, change if they are png or others


distance_matrix=np.zeros((0,2),dtype='S20, f4')


for i in range(len(allFiles)): # fron 0 to len(allFiles)-1 --> search for all images in the database
    candidate_image=os.path.join(dBPath,allFiles[i]) # get candidate
    hash_candidate= get_descriptors(candidate_image,num_max_features) # get hash of candidate
    dist = np.linalg.norm((hash_candidate-hash_query), ord=1) # compute l1 norm between hashes
    distance_matrix = np.append(distance_matrix, np.array([(allFiles[i], dist)], dtype='S20, f4')) # append candidate names and distances into a matrix
    

np.sort(distance_matrix.view('S20,f4'), order=['f1'], axis=0) # The sort matrix of distances by distances. This is the list of images in the database with the distance between the query hash and the 
# database image hash.
