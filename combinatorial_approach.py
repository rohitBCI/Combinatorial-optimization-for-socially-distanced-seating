#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
from numpy import*
import copy
np.set_printoptions(threshold=np.inf)
from itertools import combinations
from timeit import default_timer as timer

#Function to create the adjaceny matrix of the cinema layout based on social distancing
def generate_adjaceny_matrix(layout):

    adjaceny_matrix = [zeros(size)]
    for i in range (rows):
        for j in range (columns):
            array = zeros(size)
            layout_position = layout[i][j]

            #No seats affected for unavialble seat
            if layout_position == '0':
                adjaceny_matrix.append(array)

            #Seats affected for each avialble seat
            elif layout_position == '1':

                #Seat affected one space right of available seat
                if((j+1)<=(columns-1) and layout[i][j+1] == '1'):
                    array[i*columns + j + 1] = 1
                #Seat affected two spaces right of available seat
                if((j+2)<=(columns-1) and layout[i][j+2] == '1'):
                    array[i*columns + j + 2] = 1
                #Seat affected one space left of available seat
                if((j-1)>=0 and layout[i][j-1] == '1'):
                    array[i*columns + j - 1] = 1
                #Seat affected two space left of available seat
                if((j-2)>=0 and layout[i][j-2] == '1'):
                    array[i*columns + j - 2] = 1
                #Seat affected below available seat        
                if((i+1)<=(rows-1) and layout[i+1][j] == '1'):
                    array[(i+1)*columns + j] = 1
                #Seat affected above available seat
                if((i-1)>=0 and layout[i-1][j] == '1'):
                    array[(i-1)*columns + j] = 1
                #Seat affected top left of available seat        
                if((i-1)>=0 and (j-1)>=0 and layout[i-1][j-1] == '1'):
                    array[(i-1)*columns + j -1] = 1
                #Seat affected bottom left of available seat
                if((i+1)<= (rows-1) and (j-1)>=0 and layout[i+1][j-1] == '1'):
                    array[(i+1)*columns + j - 1] = 1
                #Seat affected top right of available seat
                if((i-1)>=0 and (j+1)<= (columns-1) and layout[i-1][j+1] == '1'):
                    array[(i-1)*columns + j + 1] = 1
                #Seat affected bottom right of available seat
                if((i+1)<= (rows-1) and (j+1)<= (columns-1) and layout[i+1][j+1] == '1'):
                    array[(i+1)*columns + j + 1] = 1
                    
                adjaceny_matrix.append(array)
    adjaceny_matrix = np.delete(adjaceny_matrix, 0, 0)  
    return adjaceny_matrix


#Function to create the adjaceny List (dictionary) which gives information about which seats an occupied seat will affect
def generate_adjaceny_list(adjaceny_matrix):
    array=[]
    adjaceny_list = {}
    for i in range (size):
        for j in range (size):
            if (adjaceny_matrix[i][j] == 1):
                array.append(j)
        for k in array:
            adjaceny_list.setdefault(i, []).append(k)
        array =[]
    return adjaceny_list

#Function to convert a cinema layout position to matrix position
def layout_to_matrix(layout_position):
    current_position = 0
    i = layout_position//columns
    j = layout_position%columns
    return (i,j)
            
#Function to convert a matrix position to cinema layout position
def matrix_to_layout(x,y):
    current_position=0
    current_position = columns*x +y
    return current_position

#Function to compute all possible seating arrangements for each group size
def possible_seating_arrangements(layout):

    seating_arrangements = []
    flag = 0
    for group in range(len(visitors)):
        seating_pos_group = []
        if(visitors[group] == 0):
            seating_arrangements.append([])
            continue
        if(group == 0):
            for i in range (rows):
                for j in range (columns):
                    if(layout[i][j] == '1'):
                        seating_pos_group.append(matrix_to_layout(i,j))
            seating_arrangements.append(seating_pos_group)  
        else:
            for i in range (rows):
                for j in range ((columns-group)):
                    for k in range(0, group+1):
                        #print(k)
                        if(layout[i][j+k] != '1'):
                            flag = 1
                    seats = []
                    if(flag == 0):
                        for l in range(0, group+1):
                            seats.append(matrix_to_layout(i,j+l))
                            if(seats not in seating_pos_group):
                                seating_pos_group.append(seats)
                    flag = 0
            seating_arrangements.append(seating_pos_group)
    return seating_arrangements

#Function to compute all possible seating arrangements for each group size sorted
def possible_group_arrangements_all(seating_arrangements):
    group_arrangements_all = []
    group_arrangements_adj = []
    for group in range(0,8):
        group_min_affected = [] 
        arr = []
        if(seating_arrangements[group]!=[]):
            temp = get_affected(seating_arrangements[group])
            #print(temp)
            for i in range(len(temp)):
                group_min_affected.append([len(temp[i]),i])
                group_arrangements_adj.append(temp[i])
            group_min_affected = np.array(group_min_affected)
            sorted_group_min_affected = group_min_affected[np.argsort(group_min_affected[:, 0])]
            seating_arrangements1 = []
            for i in sorted_group_min_affected[:, 1]:
                seating_arrangements1.append(seating_arrangements[group][i])

            group_affected = get_affected(seating_arrangements1)
            count = 0
            flag = 0 
            conc = []

            for i in range(len(group_affected)):
                arr.append(seating_arrangements1[i])   
        group_arrangements_all.append(arr)
    return group_arrangements_all

#Function to compute all possible affected seats for each group size sorted
def possible_group_arrangements_adj(group_arrangements_all):
    group_arrangements_adj = []
    for i in range (0,8):
        arr= []
        if(group_arrangements_all[i]!=[]):
            arr = get_affected(group_arrangements_all[i])
        group_arrangements_adj.append(arr)
    return group_arrangements_adj

#Function to find intersection between two arrays
def intersec(d):
    unq = []
    intersection = []
    for i in d:
        for j in i:
            if j in unq:
                intersection.append(j)
            else:
                unq.append(j)
    return len(intersection)

#Function to find intersection between two arrays of group size 1
def intersec1(x,y):
    c= 0 
    for i in x:
        if i in y:
            c =c+1
    return c

#Function to find intersection between two arrays
def intersec_arrays(d,e):
    unq = []
    inter = []
    inter1 = []
    for i in d:
        for j in i:
            if j in unq:
                inter.append(j)
            else:
                unq.append(j)
    for i in e:
        if i in inter:
            inter1.append(j)
    return len(inter1)
#Function to merge arrays into a single array

def merge_arrays(arrays):
    merged_array = arrays[0]
    for i in range(1,len(arrays)):
        merged_array = merged_array + arrays[i]
    merged_array = np.unique(merged_array)
    return merged_array


#Function that places all groups except groups of one in the cinema
def place_group(group):
    affected_pos = []
    seated_pos = []
    for w in range (len(group)):    #looping over the individual nodes of a signle group  position
        group_element = group[w]            #stores a signle value of a single group  position
        adj_list = adjacency_list[group_element]
        seated_pos.append(group_element)
        for q in adj_list:
            if(q not in group):
                affected_pos.append(q)
    return np.sort(np.unique(affected_pos)), np.sort(np.unique(seated_pos))

#Function that computes affected seats when placing a group
def get_affected(array):
    affected_seats=[]
    for i in array:
        affected_array=[]
        if(isinstance(i, int)):
            adj_list = adjacency_list[i]
            for k in adj_list:
                affected_array.append(k)
            affected_unique_array = np.unique(affected_array)
            affected_seats.append(affected_unique_array.tolist())
        else:
            for j in i:
                adj_list = adjacency_list[j]
                for k in adj_list:
                    affected_array.append(k)
            affected_unique_array = np.unique(affected_array)
            affected_seats.append(affected_unique_array.tolist())
    return affected_seats

#Function that find best possible combination for a particular group size
def combinations_adj(iterable, iterable_adj, r, seated_positions, blocked_positions):
    fin = []
    pool = []
    pool_adj = []
    max_indices = []
    min1 = 100
    
    pool1 = tuple(iterable)
    pool_adj1 = tuple(iterable_adj)
    for w in range(len(pool1)):
        if(intersec([pool1[w], blocked_positions]) == 0):
            pool.append(pool1[w])
            pool_adj.append(pool_adj1[w])
    
    pool = tuple(pool)
    pool_adj = tuple(pool_adj)
    n = len(pool)
    
    if r > n:
        return []
    
    indices = list(range(r))
    d = tuple(pool[i] for i in indices)
    d_adj = tuple(pool_adj[i] for i in indices)
    tmp = merge_arrays(d)
    tmp_adj = merge_arrays(d_adj)
    
    #Constraints for social distancing
    group_dist = intersec([tmp,seated_positions])
    group_dist1 = intersec(d)
    group_dist2 = intersec([tmp,blocked_positions]) 
    group_dist3 = intersec_arrays(d_adj, tmp)
    adj_dist = intersec([tmp_adj,blocked_positions])
    total_dist = len(tmp_adj) + len(blocked_positions) - adj_dist
    
    if(group_dist == 0 and group_dist1 == 0 and group_dist2 == 0 and group_dist3 == 0 and total_dist < min1):
        min_indices = indices
        min1 = total_dist
        fin =  tuple(pool[i] for i in min_indices)
    
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return fin 
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
            
        d = tuple(pool[i] for i in indices)
        d_adj = tuple(pool_adj[i] for i in indices)
        tmp = merge_arrays(d)
        tmp_adj = merge_arrays(d_adj)
        
        #Constraints for social distancing
        group_dist = intersec([tmp,seated_positions])
        group_dist1 = intersec(d)
        group_dist2 = intersec([tmp,blocked_positions])
        group_dist3 = intersec_arrays(d_adj, tmp)
        
        adj_dist = intersec([tmp_adj,blocked_positions])
        total_dist = len(tmp_adj) + len(blocked_positions) - adj_dist
        
        if(group_dist == 0 and group_dist1 == 0 and group_dist2 == 0 and group_dist3 == 0 and total_dist < min1):
            min_indices = indices
            min1 = total_dist
            fin =  tuple(pool[i] for i in min_indices)
    return fin

#Function that find best possible combination for group size 1
def combinations_adj1(iterable, iterable_adj, r, seated_positions, blocked_positions):
    fin = []
    max_indices = []
    pool = []
    pool_adj = []
    min1 = 100
    
    pool1 = tuple(iterable)
    pool_adj1 = tuple(iterable_adj)
    
    for i in range(len(pool1)):
        if pool1[i] not in blocked_positions:
            pool.append(pool1[i])
            pool_adj.append(pool_adj1[i])
            

    pool = tuple(pool)
    pool_adj = tuple(pool_adj)
    n = len(pool)
    
    if(n==0):
        return []
    if r > n:
        return []
    
    indices = list(range(r))
    d = tuple(pool[i] for i in indices)
    d_adj = tuple(pool_adj[i] for i in indices)
    d_adj = merge_arrays(d_adj)
    #Constraints for social distancing
    group_dist = intersec([d,seated_positions])
    group_dist1 = intersec([d])
    group_dist2 = intersec([d,blocked_positions]) 
    group_dist3 = intersec1(d_adj,d)

    if(group_dist == 0 and group_dist1 == 0 and group_dist2 == 0 and group_dist3 == 0):
        min_indices = indices
        fin =  tuple(pool[i] for i in min_indices)
        return fin
    
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return fin 
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        d = tuple(pool[i] for i in indices)
        d_adj = tuple(pool_adj[i] for i in indices)
        d_adj2 = merge_arrays(d_adj)
        
        #Constraints for social distancing
        group_dist = intersec([d,seated_positions])
        group_dist1 = intersec([d])
        group_dist2 = intersec([d,blocked_positions]) 
        group_dist3 = intersec1(d_adj2,d)
        
        if(group_dist == 0 and group_dist1 == 0 and group_dist2 == 0 and group_dist3 == 0):
            min_indices = indices
            fin =  tuple(pool[i] for i in min_indices)
            break
    return fin

#Function that seats groups for multiple iterations
def new_alg(visitors_copy):
    tmp1 = []
    arr1 = []
    for k in range(7,-1,-1):
        if(k==7):
            if(visitors_copy[k]!=0):
                comb = combinations_adj(group_arrangements_all[k], group_arrangements_adj[k], visitors_copy[k], [],[])
                if(comb == []):
                    return False
                c = merge_arrays(comb)
                tmp1 = c
                s,t = place_group(tmp1)
                arr1 = np.concatenate((s, t))
        elif(k!=0):
            if(visitors_copy[k]!=0):
                comb = combinations_adj(group_arrangements_all[k], group_arrangements_adj[k], visitors_copy[k], tmp1,arr1)
                if(comb == []):
                    return False
                c = merge_arrays(comb)
                tmp1 = np.concatenate((tmp1,c))
                s,t = place_group(tmp1)
                arr1 = np.concatenate((s, t))
        else:
            v = visitors_copy[k]
            if (v!=0):
                rem1 = []
                for i in total_seats:
                    if i not in arr1:
                        rem1.append(i)
                if(rem1 != []):
                    lrem1 = len(rem1)
                    if(v>lrem1):
                        v = lrem1
                comb = combinations_adj1(group_arrangements_all[k], group_arrangements_adj[k], visitors_copy[k], tmp1,arr1)
                tmp1 = np.concatenate((tmp1,comb))
                s,t = place_group(tmp1)
                arr1 = np.concatenate((s, t))
                
    return [tmp1 ,arr1]

#Function that seats groups during first iteration
def new_alg1(visitors_copy):
    tmp1 = []
    arr1 = []
    k = 7
    while(k>=0):
        if(k!=0):
            if(visitors_copy[k]!=0):
                comb = combinations_adj(group_arrangements_all[k], group_arrangements_adj[k],visitors_copy[k], tmp1,arr1)
                if(comb == []):
                    visitors_copy[k]  = visitors_copy[k] - 1
                    continue
                c = merge_arrays(comb)
                tmp1 = np.concatenate((tmp1,c))
                s,t = place_group(tmp1)
                arr1 = np.concatenate((s, t))
        else:
            v = visitors_copy[k]
            if (v!=0):
                rem1 = []
                for i in total_seats:
                    if i not in arr1:
                        rem1.append(i)
                if(rem1 != []):
                    lrem1 = len(rem1)
                    if(v>lrem1):
                        v = lrem1
                comb = combinations_adj1(group_arrangements_all[k], group_arrangements_adj[k], visitors_copy[k], tmp1,arr1)
                tmp1 = np.concatenate((tmp1,comb))
                s,t = place_group(tmp1)
                arr1 = np.concatenate((s, t))
        k = k - 1
                
    return [tmp1 ,arr1]

#Function to find all possible group permutations
def possible_group_permutations(max_seated):
    
    possible_group_values =[]
    for i in visitors:
        array=[]
        for j in range (0,i+1):
            array.append(j)
        possible_group_values.append(array)
    
    group_arrangements = []
    for i in  possible_group_values[0]:
        for i1 in  possible_group_values[1]:
            for i2 in  possible_group_values[2]:
                 for i3 in  possible_group_values[3]:
                    for i4 in  possible_group_values[4]:
                        for i5 in  possible_group_values[5]:
                             for i6 in  possible_group_values[6]:
                                for i7 in  possible_group_values[7]:
                                    sum_seated = i*1 + i1*2 + i2*3+ i3*4+ i4*5+ i5*6+ i6*7+ i7*8
                                    if(sum_seated >= max_seated):
                                        group_arrangements.append([[i,i1,i2,i3,i4,i5,i6,i7],sum_seated])
                                        
    return group_arrangements

filename = input("Enter path to input file") 

start = timer()
f = open(filename, "r")

#Extracting number of rows and columns
user_input = []
rows = int(f.readline())
columns = int(f.readline())

#Reading rest of user input
for line in f.readlines():
    line.strip()
    user_input.append(line.split())

#Extracting number of respective visitor groups
visitors = []
visitors_input = user_input[-1]
for i in visitors_input:
    visitors.append(int(i))

visitors_copy =  copy.deepcopy(visitors)
user_input = np.delete(user_input,-1, 0)
f.close()

#Extracting cinema layout as a 2D array
array_2D = []
for i in user_input:
    string = str(i)[1:-1]
    array = [(string[j:j+1]) for j in range(0, len(string), 1)] 
    array = np.delete(array, 0, 0)
    array = np.delete(array, -1, 0)
    array_2D.append(array)

cinema_layout = array_2D[:]  
cinema_layout_copy = z = copy.deepcopy(cinema_layout)
max_seated = 0
size = rows * columns

total_seats = []
for i in range(len(cinema_layout)):
    t = cinema_layout[i]
    for j in range(len(t)):
        if(t[j] == '1'):
            total_seats.append(matrix_to_layout(i,j))

adjaceny_matrix = generate_adjaceny_matrix(cinema_layout_copy)
adjacency_list = generate_adjaceny_list(adjaceny_matrix)

seating_arrangements = possible_seating_arrangements(cinema_layout_copy)

group_arrangements_all = possible_group_arrangements_all(seating_arrangements)

group_arrangements_adj = possible_group_arrangements_adj(group_arrangements_all)


final1 = new_alg1(visitors_copy)

people_seated = len(final1[0])
visitor_permutations = possible_group_permutations(people_seated)
visitor_permutations_sorted = sorted(visitor_permutations, key=lambda x:(x[1],x[0]))
visitor_permutation_sorted1 = []
for i in range(1, len(visitor_permutations_sorted )):
    t = visitor_permutations_sorted [i]
    if(t[1] > people_seated):
        visitor_permutation_sorted1.append(t)

flag = 0
if(visitor_permutation_sorted1==[]):
    flag = 0
if(visitor_permutation_sorted1!=[]):
    max_seated = people_seated
    maxi = 0
    seating = []

count = 0
import time
future  = time.perf_counter() + 20
for i in range(0,len(visitor_permutation_sorted1)):
    if(time.perf_counter() > future):
        break
    if(i==0):
        itr1_size = visitor_permutation_sorted1[i][1]
    size = visitor_permutation_sorted1[i][1]
    if(size == max_seated):
        continue
    else:
        if(size > itr1_size + 2):
            break
        vis = visitor_permutation_sorted1[i][0]
        final = new_alg(vis)
        if(final != False):
            occupied_seats = len(final[0])
            if(occupied_seats > max_seated):
                max_seated = occupied_seats
                itr1_size = occupied_seats
                flag = 1
                count = 0
                maxi = i
                seating = final[0] 
            else:
                continue
        else:
            continue
    opt_arrangement = visitor_permutation_sorted1[maxi][0]

if(flag == 0):
    seating = final1[0]
    opt_arrangement = visitors_copy
output_cinema_layout_array = copy.deepcopy(cinema_layout_copy)
people_seated = 0
for i in range(rows):
    for j in range(columns):
        m_value = matrix_to_layout(i,j)
        if(m_value in seating):
            output_cinema_layout_array[i][j] = 'x'
            people_seated = people_seated + 1

output_cinema_layout = []
for i in output_cinema_layout_array:
    t = []
    for j in i:
        t.append(j)
    output_cinema_layout.append(t)
    
f = open("output_file.txt","a")
f.write("Input layout\n")
for i in cinema_layout:
    f.write(str(i))
    f.write("\n")
f.write("Visitor Input :")
f.write(str(visitors))
f.write("\n")
f.write("\n---Output Genertaed---")
f.write("\nTime taken for execution : ")
f.write(str(timer()-start))
f.write("\nOptimal seating size : ")
f.write(str(people_seated))
f.write("\nOptimal seating layout\n")
for i in output_cinema_layout:
    f.write(str(i))
    f.write("\n")
f.write("\n")
f.close()

