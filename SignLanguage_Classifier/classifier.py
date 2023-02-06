import cv2
import numpy as np
import os

debug = False

def preprocess(im_in):

    # Threshold
    _, im_th = cv2.threshold(im_in, 240, 255, cv2.THRESH_TOZERO_INV)  # >220 goes to 0, <220 goes to 255 #WAS ORIGINALLY 120
    im_floodfill = im_th.copy()

    # Black image with same number of pixels
    h = im_th.shape[0]
    w = im_th.shape[1]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # All zero-valued pixels in the same connected component as the seed point of the mask are replaced with 255
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # bitwise NOT
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # bitwise OR
    im_out = cv2.bitwise_or(im_th, im_floodfill_inv, mask)

    # bitwise NOT
    final = cv2.bitwise_not(im_out)

    return final


def find_min_rect_coordinates(img):
    # create a numpy array with the image
    og_image = np.array(img)

    # obtain all the (x,y) points where the image is black
    indices = np.where(og_image == [0])

    # save the coordinates in one list
    coordinates = zip(indices[1], indices[0])  # coordinates are flipped here

    max_x = (0, 0)
    min_y = (0, 1000)
    min_x = (1000, 0)
    max_y = (0, 0)

    for x, y in coordinates:

        if x > max_x[0]:
            max_x = (x, y)

        if x < min_x[0]:
            min_x = (x, y)

        if y > max_y[1]:
            max_y = (x, y)
        if y < min_y[1]:
            min_y = (x, y)

    y = min_y[1]
    x = min_x[0]
    h = max_y[1] - min_y[1]
    w = max_x[0] - min_x[0]

    # Find the aspect ratio
    aspect_ratio = float(w) / float(h)

    return x, y, w, h  # where x,y is the top starting point and h is the height and w is the width


def find_features(x, y, w, h, thresholded):
    og_image = thresholded
    aspect_ratio = float(w) / float(h)

    # Left/Right hand side of the image
    lfs_img = og_image[y:y + h, x:x + w // 2]

    val = w / 2

    if val.is_integer():
        rhs_img = og_image[y:y + h, x + w // 2: x + w]
    else:
        rhs_img = og_image[y:y + h, x + w // 2 + 1: x + w]

    # Left/Right hand side, number of black pixels
    num_of_black_rhs = np.sum(rhs_img == 0)
    num_of_black_lfs = np.sum(lfs_img == 0)
    # figure out the rectangle area
    a_rect = w * h
    lfs_ratio = num_of_black_lfs / a_rect
    rhs_ratio = num_of_black_rhs / a_rect

    return aspect_ratio, lfs_ratio, rhs_ratio


#dmin works with 3D objects
def dmin(A,B):
    distance = []
    for coordinate1 in A:
        for coordinate2 in B:
            x = (coordinate2[0] - coordinate1[0])**2
            y = (coordinate2[1] - coordinate1[1])**2
            z = (coordinate2[2] - coordinate1[2])**2
            distance.append((x+y+z)**0.5)

    print('the minimum distance is',min(distance))
    return min(distance)

#this creates a list with 10 elements, each element is a cluster of points corresponding to a class of images
#  points = [[(x1,y1,z1)...(xn,yn,zn)],[(x1,y1,z1)...(xn,yn,zn)]] this 2 clusters with n points in each one
def create_train_clusters(dir_train, bonus): # TODO: DONE

    cluster0 = []
    cluster1 = []
    cluster2 = []
    cluster3 = []
    cluster4 = []
    cluster5 = []
    cluster6 = []
    cluster7 = []
    cluster8 = []
    cluster9 = []
    clusters = [cluster0,cluster1,cluster2,cluster3,cluster4,cluster5,cluster6,cluster7,cluster8,cluster9]          #ADJUST THIS WHEN ADDING NEW CLUSTERS

    if not bonus:
        signal_0 = [0.712, 0.354, 0.249]
        signal_1 = [0.466, 0.247, 0.339]
        signal_2 = [0.44, 0.319, 0.323]
        signal_3 = [0.671, 0.327, 0.17]
        signal_4 = [0.564, 0.218, 0.368]
        signal_5 = [0.858, 0.234, 0.202]
        signal_6 = [0.431, 0.335, 0.378]
        signal_7 = [0.551, 0.157, 0.405]
        signal_8 = [0.591, 0.145, 0.386]
        signal_9 = [0.622, 0.174, 0.355]
        clusters = [signal_0,signal_1,signal_2,signal_3,signal_4,signal_5,signal_6,signal_7,signal_8,signal_9]

    i = -1

    if bonus:
        #bonus variable will determine whether the bonus question is going to be examined or not
        for _, folders, _ in os.walk(dir_train):

            for folder in folders: #folders = {0,1}
                i += 1

                for image_name in os.listdir(os.path.join(dir_train,folder)): #images = {1,2,...,11}
                    if image_name != '.DS_Store':  # this is for a file that is in each file

                        path = os.path.join(dir_train,folder,image_name)
                        x, y, z = create_feature_vector(path)
                        print('FOLDER',i,' IMAGE NAME:',image_name,'x: ',x,'y: ',y,'z: ',z)
                        clusters[i].append([x,y,z])

    return clusters


def find_accuracy(actual,predicted):
    print('actual',actual)
    print('predicted',predicted)

    n = len(actual)
    correct = 0

    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1

    print('correct',correct)
    print('n',n)
    return (correct/n) * 100


def predict(test_clusters, train_clusters):

    dmin_measures = []
    true_class = -1

    actual = []         #These 2 values will be used to determine the accuracy of the model
    predicted = []

    if not bonus:
        true_class = 8 #TODO: change this later

    #The test inages are already sroted by class, so we access them by class first
    for img_test_class in test_clusters: #img_test_class = 0 or 1 list
        true_class += 1
        for point in img_test_class:

            #for every cluster in the training set, calculate how far our point is from it
            for i in range(len(train_clusters)):
                print('NOW EXAMNING THE DISTANCE OF CLUSTER', i, 'FROM THE TRAINING SET')
                print('POINT',point)

                #bandaid -solution

                if (bonus):
                    dist = dmin([point],train_clusters[i])
                elif not bonus:
                    dist = dmin([point], [train_clusters[i]])
                dmin_measures.append(dist)

            # TODO: FOR ASSIGNMENT PRINT OUT DMIN_MEASURES HERE
            if not bonus:
                print(dmin_measures)

            #find cluster with the smallest distance
            m = min(dmin_measures)
            predicted_class = dmin_measures.index(m)
            print('THE PREDICTED CLASS IS:', predicted_class)
            print('THE TRUE CLASS IS:', true_class)
            actual.append(true_class)
            predicted.append(predicted_class)


            #reset the dmeasures array
            dmin_measures = []

    acu = find_accuracy(actual,predicted)
    print('THE ACCURACY OF THE MODEL IS.....',acu)


#This creates a list of points that individually belong to an image, the list of points is actually grouped by class to make accuracy computation simple
# points = [[(x1,y1,z1)...(xn,yn,zn)],[(x1,y1,z1)...(xn,yn,zn)]] this 2 classes with n points in each one

def create_test_clusters(dir_test, bonus):

    points0 = []
    points1 = []
    points2 = []
    points3 = []
    points4 = []
    points6 = []
    points7 = []
    points8 = []
    points5 = []
    points9 = []
    points = [points0,points1,points2,points3,points4,points5,points6,points7,points8,points9]          #ADJUST THIS WHEN ADDING NEW CLUSTERS

    if not bonus:
        path = r'/Users/divya/PycharmProjects/CP467/compare_to/horns.png'
        #create the feature vector for that image given in the example
        x, y, z = create_feature_vector(path)
        points = [[x,y,z]]

    i = -1 #counter variable

    if bonus:
        #bonus variable will determine whether the bonus question is going to be examined or not
        for _, folders, _ in os.walk(dir_test):

            for folder in folders: #folders = {0,1}
                i += 1
                print('IN THIS FOLDER ********************************************')

                for image_name in os.listdir(os.path.join(dir_test,folder)): #images = {1,2,...,11}
                    if image_name != '.DS_Store':  # this is for a file that is in each file

                        path = os.path.join(dir_test,folder,image_name)
                        x, y, z = create_feature_vector(path)
                        print('FOLDER',i,' IMAGE NAME:',image_name,'x: ',x,'y: ',y,'z: ',z)
                        points[i].append([x,y,z])

    return points

def create_feature_vector(im_dir):
    # Read the image
    im_in = cv2.imread(im_dir, cv2.IMREAD_GRAYSCALE)

    if debug:
        cv2.imshow("og", im_in)
        #cv2.waitKey(0)

    # Threshold
    thresholded = preprocess(im_in)

    if debug:
        cv2.imshow('thresholded',thresholded)
        cv2.waitKey(0)

    # Find the rectangle
    x, y, w, h = find_min_rect_coordinates(thresholded)

    # Find the features
    x_fe, y_fe, z_fe = find_features(x, y, w, h, thresholded)

    return x_fe,y_fe,z_fe



bonus = True
base_path_train = r'images/train'#r'/Users/divya/PycharmProjects/CP467/handwritten_letters/train'
base_path_test = r'images/test'#r'/Users/divya/PycharmProjects/CP467/handwritten_letters/test'


#create the clusters for 10 images
print('-------------------------------------------------------------TRAIN')
clus_train = create_train_clusters(base_path_train,bonus)
print('the length of train',len(clus_train))
#create the test clusters
print('-------------------------------------------------------------TEST')
clus_test = create_test_clusters(base_path_test,bonus) #Does not work when =False
print('the length of test',len(clus_test))

if(bonus):
    predict(clus_test, clus_train)
else:
    predict([clus_test], clus_train)


# TESTING
#this is 1 test image:
#t_im = r'/Users/divya/PycharmProjects/CP467/handwritten_letters/test/1/1_7.png'
#test_cluster = create_feature_vector(t_im)

# #compare to cluster 0
# distance_0 = dmin([test_cluster], clus_train[0])
# #compare to cluster 1
# distance_1 = dmin([test_cluster], clus_train[1])
#
# print('distance 0:', distance_0)
# print('distance 1:', distance_1)
# if distance_0 < distance_1:
#     print('test image is closer to class 0')
# else:
#     print('test image is closer to class 1')

#display_table(np.round_(x_f, decimals=3), np.round_(y_f, decimals=3), np.round_(z_f, decimals=3))
