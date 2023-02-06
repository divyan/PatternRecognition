import cv2

distance = []
def dmin(A,B):
    for coordinate1 in A:
        for coordinate2 in B:
            x = (coordinate2[0] - coordinate1[0])**2
            y = (coordinate2[1] - coordinate1[1])**2
            distance.append((x+y)**0.5)
    print(min(distance))
    
def dmax(A,B):
    for coordinate1 in A:
        for coordinate2 in B:
            x = (coordinate2[0] - coordinate1[0])**2
            y = (coordinate2[1] - coordinate1[1])**2
            distance.append((x+y)**0.5)
    print(max(distance))
    
def davg(A,B):
    sum_distance = 0
    i = len(A)
    j = len(B)
    for coordinate1 in A:
        for coordinate2 in B:
            sum_distance += ((coordinate2[0] - coordinate1[0])**2+(coordinate2[1] - coordinate1[1])**2)**0.5
    print(sum_distance/(i*j))
    
def find_max_min(points,maxmim_val,minimum_val):
    maximum = maxmim_val
    minimum = minimum_val

    for coordinate in points:
        #max
        if coordinate[0] > maximum:
            maximum = coordinate[0]
        if coordinate[1] > maximum:
            maximum = coordinate[1]
        #min
        if coordinate[0] < minimum:
            minimum = coordinate[0]
        if coordinate[1] < minimum:
            minimum = coordinate[1]

    return(maximum,minimum)
def dmean(A,B):

    maximum, minimum = find_max_min(A, -99999, 99999)
    maximum, minimum = find_max_min(B, maximum, minimum)

    #calcualte the mean
    mean_val = (( (minimum/2)**2 + (maximum/2)**2 )**(1/2) )/ len(A)*len(B)

    print(mean_val)

A = [ (8,8),(9,9)] # [(1,1), (1,2)]
B = [(4,6),(5,8)] # [(2,1),(3,1)]
dmin(A,B)
dmax(A,B)
davg(A,B)
dmean(A,B)
