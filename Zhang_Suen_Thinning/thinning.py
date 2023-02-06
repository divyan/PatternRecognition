import cv2
import numpy as np
import sys

'''
find_neighbours : finds 8 point neighbourhood 

Input:
x  - int: row index of P1 
y  - int: col index of P1
IT - list: binary image

Output:
neighbourhood - list: P2-P9 in clockwise direction 

P9  -  P2  - P3
P8  - *P1* - P4 
P7  -  P6  - P5  
'''
def find_neighbourhood(row,col,IT):
    P2 = IT[row - 1][col]
    P3 = IT[row - 1][col + 1]
    P4 = IT[row][col+1]
    P5 = IT[row + 1][col + 1]
    P6 = IT[row + 1][col]
    P7 = IT[row + 1][col - 1]
    P8 = IT[row][col - 1]
    P9 = IT[row - 1][col - 1]

    neighbourhood = [P2, P3, P4, P5, P6, P7, P8, P9]
    return neighbourhood

'''
CONDITION A)
find_nonzero_nbs : finds the number of non_zero magnitude neighbours

Input:
neighbourhood   - list: P2-P9 in clockwise direction 

Output:
n_nonzero - int: number of 1's in ordered set P2-P9 
'''
def find_nonzero_nbs(neighbourhood):

    n_nonzero = sum(neighbourhood)

    return n_nonzero


'''
CONDITION B)
find_transitions : finds the number of transitions

Input:
neighbourhood   - list: P2-P9 in clockwise direction 

Output:
n_transitions   - int: number of 01 patterns in ordered set P2-P9 
'''
def find_transitions(neighbourhood):
    nd = neighbourhood  # for simplicity
    n_transitions = 0
    prev = 0
    curr = 0

    for i in range(len(nd)):
        curr = nd[i]

        if prev == 0 and curr == 1:
            n_transitions += 1
        prev = curr

    return n_transitions

'''
ITER 1 & 2 CONDITION C and D)
find_product1 : finds the product of P2, P4, P6  and P4, P6, P8 for sub-iteration 1 
                or, P2, P4, P8, and P2, P4, P8 for sub-iteration 2 

Input:
neighbourhood       - list: P2-P9 in clockwise direction
subiter             - int: specifies which iteration the algorithm is on  

Output:
product1, product 2 - int, int:  product of North, East and South points, 
                                 products of the East, South and West points
'''
def find_product(neighbourhood, subiter):

    if subiter == 1:
        product1 = neighbourhood[0] * neighbourhood[2] * neighbourhood[4]
        product2 = neighbourhood[2] * neighbourhood[4] * neighbourhood[6]
    elif subiter == 2:
        product1 = neighbourhood[0] * neighbourhood[2] * neighbourhood[6]
        product2 = neighbourhood[0] * neighbourhood[4] * neighbourhood[6]

    return product1, product2

def delete_p(to_be_deleted, img):
    for pair in to_be_deleted:
        x = pair[0]
        y = pair[1]

        img[x,y] = 0

    return img

'''
sub_iter1 : deletes pixels ...

Input:
img   - list: binary image 

Output:
c   - int: 1 if change occured, 0 if change does not occur 
'''
def sub_iter1(img):
    print('sub iteration 1')
    to_be_deleted = []
    c = 0               # whether changes are being made or not
    IT = img            # already binary here
    rows , cols = IT.shape

    for i in range(1,rows-1):
        for j in range(1,cols-1):
            nbhd = find_neighbourhood(i, j, IT)

            # check conditions
            a = 2 <= find_nonzero_nbs(nbhd) <= 6
            b = find_transitions(nbhd) == 1
            c_d = find_product(nbhd, 1) == (0, 0)

            conditions = a * b * c_d  # AND GATE with all the conditions

            if IT[i, j] == 1 and conditions:
                to_be_deleted.append((i, j))  # store pixel for deletion
                c = 1

    new_img = delete_p(to_be_deleted, IT)

    return new_img, c

def sub_iter2(img):
    print('sub iteration 2')
    to_be_deleted = []
    c = 0               # whether changes are being made or not
    IT = img            # already binary here
    rows , cols = IT.shape

    for i in range(1,rows-1):
        for j in range(1,cols-1):
            nbhd = find_neighbourhood(i, j, IT)

            # check conditions
            a = 2 <= find_nonzero_nbs(nbhd) <= 6
            b = find_transitions(nbhd) == 1
            c_d = find_product(nbhd, 2) == (0, 0)

            conditions = a * b * c_d  # AND GATE with all the conditions

            if IT[i, j] == 1 and conditions:
                to_be_deleted.append((i, j))  # store pixel for deletion
                c = 1

    # delete points
    new_img = delete_p(to_be_deleted, IT)

    return new_img, c


def z_s_algo(img_gray):

    img = cv2.threshold(img_gray, 170, 1, cv2.THRESH_BINARY_INV)[1] #[0] = thresh, [1] = img cv2.threshold(img_gray, 170, 1, cv2.THRESH_BINARY)[1]
    print(img.shape)
    i = 0

    c1 = 1
    c2 = 1

    while c1 or c2:
        print('iteration', i)
        new_img, c1 = sub_iter1(img)
        final_img, c2 = sub_iter2(new_img)
        i+=1

    return final_img


# img = [ [0,1,0] ,
#         [0,1,1],
#         [0,1,0] ]
# n = find_neighbourhood(1,1,img)
# trans = find_transitions(n)
# p1_1, p1_2 = find_product(n,1)
# p2_1, p2_2 = find_product(n,2)
# nz = find_nonzero_nbs(n)

img_gray = cv2.imread(r'grey.jpg', cv2.IMREAD_GRAYSCALE)  # make this binary if not already

print(img_gray.shape)
thinned = z_s_algo(img_gray)

#0s go  to 255
#255s goes to 0
thinned[thinned == 0] = 255
thinned[thinned == 1] = 0
rgb_img = cv2.cvtColor(thinned, cv2.COLOR_GRAY2RGB)
cv2.imwrite('final_grey.png',rgb_img)

