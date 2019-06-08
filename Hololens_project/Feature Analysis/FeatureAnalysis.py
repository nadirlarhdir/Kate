
# coding: utf-8

import json
import numpy as np
import pandas as pd
import glob
import seaborn as sns
import os, json
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Pour faire des graphiques 3D
from sklearn.neighbors import KNeighborsClassifier


#Enter the file path where all the JSON files are located

path_to_json = '/kate_data'

game_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

#Enter Players to Study
players_to_study = [8,9]
#Assumption : By mentioning 1,2,3, etc. as players, we assume that the corresponding json files are 1.json, 2.json, 3.json, etc.. 


def norm(vect):
    sum = 0
    
    for el in vect:
        sum += el**2
    
    return np.sqrt(sum)


# Let's create a Panda's dataFrame with position, time, rotatio, BPM  of each frame of the game and a second Dataframe with the balloons gathering data



def create_df_hand(game_file):
    with open(game_file) as json_file:
        data = json.load(json_file)
        
    df_game = pd.DataFrame(data['datasList'][0]['listLevelDatas'][0]['userDatas'])
    for i in range(1,len(data['datasList'][0]['listLevelDatas'])):
        df_game = pd.concat([df_game, pd.DataFrame(data['datasList'][0]['listLevelDatas'][i]['userDatas'])])
        
    
    
    #getting rid of the timeStamp's zero
    df_game = df_game[df_game['timeStamp']>0]
    
    #reset index after having got rid of the timeStamp zeros
    df_game = df_game.reset_index(drop = True) 
    
    #let's create three new columns, each one with one coordinate for df_game:
    #If they show later to be useless, we supprime these lines to get rid of them
    position =  df_game['headPos'].apply(pd.Series)
    df_game = pd.concat([df_game, position], axis=1)
    
    #Drops the duplicated rows in the Hololens DataSet
    indexes_to_drop = []
    for index, row in df_game.iterrows():
        if index != 0:
            if df_game.loc[index, 'timeStamp'] == df_game.loc[index-1, 'timeStamp']:
                indexes_to_drop.append(index)
    #print('length indexes_to_drop:', len(indexes_to_drop))
    df_game.drop(df_game.index[indexes_to_drop], inplace=True)
    df_game = df_game.reset_index(drop = True)
    

    #Fixes the bug in the timeOfDestroy and timeOfSpawn that came with Hololens Data (values were reseting)
    for index, row in df_game.iterrows():
        if index != 0:
            if df_game.loc[index, 'timeStamp'] < df_game.loc[index-1, 'timeStamp']:
                for idx in range(index,len(df_game)):
                    df_game.at[idx, 'timeStamp'] = df_game.at[idx, 'timeStamp']  +df_game.at[index-1, 'timeStamp']
                    
    #Here we create a column withe a 5-element tuple: (x,y,z,t, rotation) for each dataframe
    df_game['head_positions'] = df_game[['x', 'y', 'z', 'timeStamp', 'headRotationY']].apply(lambda x: tuple(x), axis=1)
                    
    
    
    return df_game

def create_df_balloon(game_file):
        
    with open(game_file) as json_file:
        data = json.load(json_file)
        
        
    df_balloon = pd.DataFrame(data['datasList'][0]['listLevelDatas'][0]['listBalloonDatas'])
    for i in range(1,len(data['datasList'][0]['listLevelDatas'])):
        df_balloon = pd.concat([df_balloon, pd.DataFrame(data['datasList'][0]['listLevelDatas'][i]['listBalloonDatas'])])
        
    df_balloon = df_balloon.reset_index(drop = True)
    
    #Drops the duplicated rows in the Hololens DataSet
    indexes_to_drop = []
    for index, row in df_balloon.iterrows():
        if index != 0:
            if df_balloon.loc[index, 'timeOfSpawn'] == df_balloon.loc[index-1, 'timeOfSpawn']:
                indexes_to_drop.append(index)
    df_balloon.drop(df_balloon.index[indexes_to_drop], inplace=True)
    
    df_balloon = df_balloon.reset_index(drop = True)
    
    #this part of the code fixes the bug in the timeOfDestroy and timeOfSpawn that came with Hololens Data (values were reseting)
    for index, row in df_balloon.iterrows():
        if index != 0:
            if df_balloon.loc[index, 'timeOfDestroy'] < df_balloon.loc[index-1, 'timeOfDestroy']:
                for idx in range(index,len(df_balloon)):
                    df_balloon.at[idx, 'timeOfDestroy'] = df_balloon.at[idx, 'timeOfDestroy']  + df_balloon.at[index-1, 'timeOfDestroy']
                    df_balloon.at[idx, 'timeOfSpawn'] = df_balloon.at[idx, 'timeOfSpawn'] + df_balloon.at[index-1, 'timeOfSpawn']
    
    return df_balloon

#df_balloon = create_df_balloon('./Hololens_data/p8.json')

#df_game = create_df_hand('./Hololens_data/p8.json')


# * The function `hand_positions` extracts the positions of the right hand along with the time corresponding to those positions. It returns an array of shape [(x, y, z, t)] (length number_of_position, with 4 elements arrays representing (x, y, z, t)).


def hand_positions(game_file):
    return list(create_df_hand(game_file)['head_positions'])


#traj = hand_positions('./Hololens_data/p1.json')
#orientation(traj[0][0], traj[1][0], traj[0][1], traj[1][1], traj[0][2], traj[1][2])

#hand_positions('./Hololens_data/p1.json')


# * The function `bubble_pop` extracts the time of each game event corresponding to the pop of a bubble by the player. It returns an array of shape [t] (length number_of_bubble_poped).


def bubble_pop(game_file):
    return list(create_df_balloon(game_file)['timeOfDestroy'])

#bubble_pop('./Hololens_data/p1.json')


# # Extraction of sub-trajectories & features
# The function `sub_trajectories` returns an array of shape [[*[(x,y,t),(x,y,t),...]*, for each bubble in wave], for each wave]. To access all positions and time of the trajectory between the *i* and *i+1* bubble of the *n* wave : *sub_trajectories[n-1][i]*.

def sub_trajectories(game_file):
    hand_position = hand_positions(game_file)
    #bubble_pop_time = bubble_pop_clean(game_file)
    bubble_pop_time = bubble_pop(game_file)
    
    th = hand_position[0][3] #change to 3 because we have one more dimension
    
    sub_traj=[]
    
    nb_waves = len(bubble_pop_time)//5
    i=0 #loop count for waves
    k=0 #loop count for hand positions
    while i<nb_waves :
        sub_traj.append([])
        j=0 #loop count for bubbles
        while j<5:
            sub_traj[i].append([])
            t = bubble_pop_time[j+5*i] #the time the bubble was gathered
            while th < t:
                sub_traj[i][j].append(hand_position[k]) #appends the position of the hand and the corresponding time
                k+=1	
                th = hand_position[k][3]
            j+=1
        i+=1
    
    return np.array(sub_traj)


#print(sub_trajectories('./Hololens_data/p1.json')[0][0][1])


# We define some functions to extract interesting features from trajectories. We first look for Static features : 
# * `length` returns the length of the trajectory *traj*
# * `barycenter` returns the barycenter of the trajectory *traj* in shape (x,y)
# * `location` returns the average distance of each point to the barycenter of the trajectory *traj*
# * `location_max` returns the maximum distance between a point of the trajectory and the barycenter of this trajectory
# * `orientation` returns the angle between points the line between *(x1, y1)* and *(x2, y2)* and the horizontal axis (in degrees)
# * `orientation_feat` returns the preceeding feature for the first two points and the last two points of the trajectory *traj*
# * `nb_turns` returns the number of turns in the trajectory *traj*, where a turn is detected if the orientation between two consecutive couples of points varies of more than *limit_angle*

def length(traj):
    l = 0
    
    for i in range(len(traj)-1):
        #l += np.sqrt((traj[i+1][0]-traj[i][0])**2 + (traj[i+1][1]-traj[i][1])**2) 
        l += np.sqrt((traj[i+1][0]-traj[i][0])**2 + (traj[i+1][1]-traj[i][1])**2+(traj[i+1][2]-traj[i][2])**2)
    
    return l

def barycenter(traj):
    x = 0
    y = 0
    z = 0
    n = len(traj)
    
    for i in range(n):
        x += traj[i][0]
        y += traj[i][1]
        z += traj[i][2]
    
    if n>0:
        return (x/n, y/n, z/n) #(x/n, y/n, z/n)
    else:
        return (0,0,0) #(0,0,0)

def location(traj):
    loc_avg = 0
    n = len(traj)
    p = barycenter(traj)
    
    for i in range(n):
        #loc_avg += np.sqrt((traj[i][0] - p[0])**2 + (traj[i][1] - p[1])**2) 
        loc_avg += np.sqrt((traj[i][0] - p[0])**2 + (traj[i][1] - p[1])**2+(traj[i][2]-p[2])**2)
        
    return loc_avg/n

def location_max(traj):
    n = len(traj)
    p = barycenter(traj)
    if n>0:
        l_max = np.max([np.sqrt((traj[i][0] - p[0])**2 + (traj[i][1] - p[1])**2+(traj[i][2]-p[2])**2) for i in range(n)])
        #l_max = np.max([np.sqrt((traj[i][0] - p[0])**2 + (traj[i][1] - p[1])**2) for i in range(n)]) 
        return l_max
    else:
        return 0

    
def orientation(x1, x2 , y1, y2, z1, z2):
    #if x2 == x1 and y2>=y1:
    #    return 90
    #elif x2 == x1 and y2<=y1:
    #    return -90
    #else:
    #    return np.arctan((y2 - y1)/(x2 - x1)) * (180/np.pi) #in degree
    if x2-x1<0:
        return [np.arctan((y2 - y1)/(x2 - x1)) * (180/np.pi),np.arctan((z2 - z1)/(x2 - x1)) * (180/np.pi)+180] #in degree
    elif z2-z1<0 and x2-x1>0:
        return [np.arctan((y2 - y1)/(x2 - x1)) * (180/np.pi),np.arctan((z2 - z1)/(x2 - x1)) * (180/np.pi)] #in degree
    if x2 == x1 and y2>=y1 and z2==z1:
        return [90,0]
    elif x2 == x1 and y2<=y1 and z2==z1:
        return [-90,0]
    elif x2-x1>0 and z2-z1>=0:
        return [np.arctan((y2 - y1)/(x2 - x1)) * (180/np.pi),np.arctan((z2 - z1)/(x2 - x1)) * (180/np.pi)+180] #in degree

def orientation_feat(traj):
    n = len(traj)
    if n>1:
        #ts = orientation(traj[0][0], traj[1][0], traj[0][1], traj[1][1])
        #te = orientation(traj[-2][0], traj[-1][0], traj[-2][1], traj[-1][1])
        ts = orientation(traj[0][0], traj[1][0], traj[0][1], traj[1][1], traj[0][2], traj[1][2])
        te = orientation(traj[-2][0], traj[-1][0], traj[-2][1], traj[-1][1], traj[-2][2], traj[-1][2])

        return (ts, te)
    else:
        #return (0,0)
        return ([0,0],[0,0])

def nb_turns(traj, limit_angle):
    nb_turns = 0
    n=len(traj)
    
    for i in range(n-2):
        if(np.abs(orientation(traj[i][0], traj[i+1][0], traj[i][1], traj[i+1][1], traj[i][2], traj[i+1][2])[0] - orientation(traj[i+1][0], traj[i+2][0], traj[i+1][1], traj[i+2][1], traj[i+1][2], traj[i+2][2])[0]) > limit_angle):
            nb_turns += 1
        #if(np.abs(orientation(traj[i][0], traj[i+1][0], traj[i][1], traj[i+1][1]) - orientation(traj[i+1][0], traj[i+2][0], traj[i+1][1], traj[i+2][1])) > limit_angle):
        #    nb_turns += 1
    
    return nb_turns


# We then define dynamic features:
# * `velocity` returns the list of the point to point velocities over the whole trajectory *traj*
# * `velocity_avg` returns the average velocity over the trajectory *traj*
# * `velocity_max` returns the greatest velocity over the trajectory *traj*
# * `velocity_min` returns the lowest velocity over the trajectory *traj*
# * `nb_vmin` returns the number of local minimum of velocity
# * `nb_vmax` returns the number of local maximum of velocity

def velocity(traj):
    velocity = []
    
    for i in range(len(traj) - 1):
        #v = norm(np.array(traj)[i+1][:2] - np.array(traj)[i][:2]) / (np.array(traj)[i+1][3] - np.array(traj)[i][3])
        v = norm(np.array(traj)[i+1][:3] - np.array(traj)[i][:3]) / (np.array(traj)[i+1][3] - np.array(traj)[i][3])
        velocity.append(v)
        
    return np.array(velocity)

def angular_speed(traj):
    angular_speed = []
    for i in range(len(traj) - 1):
        w = (np.array(traj)[i+1][4] - np.array(traj)[i][4]) / (np.array(traj)[i+1][3] - np.array(traj)[i][3])
        angular_speed.append(w)
        
    return np.array(angular_speed)

def velocity_avg(traj):
    v_avg = 0
    n = len(traj)
    if n>1:
        v_list = velocity(traj)

        for i in range(n-1):
            v_avg += v_list[i]

        return v_avg/(n-1)
    else:
        return 0

def angular_speed_avg(traj):
    w_avg = 0
    n = len(traj)
    if n>1:
        w_list = angular_speed(traj)

        for i in range(n-1):
            w_avg += w_list[i]

        return w_avg/(n-1)
    else:
        return 0 

def velocity_max(traj):
    if len(traj)>1:
        return np.max(angular_speed(traj))
    else:
        return 0

def angular_speed_max(traj):
    if len(traj)>1:
        return np.max(angular_speed(traj))
    else:
        return 0

def velocity_min(traj):
    if len(traj)>1:
        return np.min(velocity(traj))
    else:
        return 0

def angular_speed_min(traj):
    if len(traj)>1:
        return np.min(angular_speed(traj))
    else:
        return 0

def nb_vmin(traj):
    nb = 0
    v_list = velocity(traj)
    
    for i in range(1,len(v_list)-1):
        if v_list[i]<v_list[i+1] and v_list[i]<v_list[i-1]:
            nb += 1
    
    return nb

def nb_wmin(traj):
    nb = 0
    w_list = angular_speed(traj)
    
    for i in range(1,len(w_list)-1):
        if w_list[i]<w_list[i+1] and w_list[i]<w_list[i-1]:
            nb += 1
    
    return nb    

def nb_vmax(traj):
    nb = 0
    v_list = velocity(traj)
    
    for i in range(1,len(v_list)-1):
        if v_list[i]>v_list[i+1] and v_list[i]>v_list[i-1]:
            nb += 1
    
    return nb


def nb_wmax(traj):
    nb = 0
    w_list = angular_speed(traj)
    
    for i in range(1,len(w_list)-1):
        if w_list[i]>w_list[i+1] and w_list[i]>w_list[i-1]:
            nb += 1
    
    return nb


# The function `feature_vector` extracts features from the trajectory in argument *traj = [(x,y,z)]*

def bucketize_nb_turns(nb_turn):
    if nb_turn == 0:
        return [1, 0, 0, 0]
    elif nb_turn == 1:
        return [0, 1, 0, 0]
    elif nb_turn < 4: # 2 ou 3
        return [0, 0, 1, 0]
    else:
        return [0, 0, 0, 1] #4 ou plus

def bucketize_nb_v(nb_v):
    if nb_v < 2:
        return [1, 0, 0, 0]
    elif nb_v < 4: # 2 ou 3
        return [0, 1, 0, 0]
    elif nb_v < 6: # 4 ou 5
        return [0, 0, 1, 0]
    else:
        return [0, 0, 0, 1] # 6 ou plus


#         x_list = [tuple_trajectory[0] for tuple_trajectory in  listetot]
#         x_min = min(x_list)
#         x_max = max(x_list)
#         y_list = [tuple_trajectory[1] for tuple_trajectory in  listetot]
#         y_min = min(y_list)
#         y_max = max(y_list)
#         game_area[0] = (x_max - x_min)
#         game_area[1] = (y_max - y_min)

def feature_vector(traj, playerID, ballon, game_area, time, limit_angle=0.25, Listefeatures=["nb_wmax", "nb_wmin", "angular_speed_min","angular_speed_max", "angular_speed_avg", "dist/diag","game area","barycenter distance","angles","nb turns","velocity average","velocity min","velocity max","number of mins","number of maxs"]):    
    diag = np.sqrt(game_area[0]**2 + game_area[1]**2)
    if len(traj)==0:
        return []
    if time=='wave':
        listetot=[]
        for k in range(len(traj)):
            listetot+=traj[k] 
        feature_vector = [playerID] 
        dist=length(listetot)
        bc=barycenter(listetot)
        if "nb_wmax" in Listefeatures:
            feature_vector.append(nb_wmax(listetot)) #feature 0
        if "nb_wmin" in Listefeatures:
            feature_vector.append(nb_wmin(listetot)) #feature 1
        if "angular_speed_min" in Listefeatures:
            feature_vector.append(angular_speed_min(listetot)) #2
        if "angular_speed_max" in Listefeatures:
            feature_vector.append(angular_speed_max(listetot)) #3
        if "angular_speed_avg" in Listefeatures:
            feature_vector.append(angular_speed_avg(listetot)) #4
        if "dist/diag" in Listefeatures:
            feature_vector.append(dist/diag) #5
        if "game area" in Listefeatures:
            feature_vector.append(np.float64(0.5 + bc[0] / game_area[0])) #6 # between 0 and 1
            feature_vector.append(np.float64(0.5 + bc[1] / game_area[1])) #7
        if location_max(listetot) == 0 and "barycenter distance" in Listefeatures:
            feature_vector.append(np.float64(0)) #8
        elif "barycenter distance" in Listefeatures:
            feature_vector.append(location(listetot)/location_max(listetot)) #8
        angles = 0.5 + np.array(orientation_feat(listetot)) / 180 # between 0 and 1
        if "angles" in Listefeatures:
            feature_vector.append(angles[0][0]) #first orientation of traj #9
            feature_vector.append(angles[0][1]) #10
            feature_vector.append(angles[1][1])#last orientation of traj #11
            feature_vector.append(angles[1][1]) #12
        if "nb turns" in Listefeatures:
            feature_vector.append(nb_turns(listetot, limit_angle)) #13
        if "velocity average" in Listefeatures:
            feature_vector.append(velocity_avg(listetot)) #14
        if "velocity min" in Listefeatures:
            feature_vector.append(velocity_min(listetot)) #15
        if "velocity max" in Listefeatures:
            feature_vector.append(velocity_max(listetot)) #16
        if "number of mins" in Listefeatures:
            feature_vector.append(nb_vmin(listetot))
        if "number of maxs" in Listefeatures:
            feature_vector.append(nb_vmax(listetot))
        return feature_vector
    if time=='pop':
        listetot=[]
        for k in range(len(traj)):
            listetot+=traj[k]
        if len(listetot)==0:
            return []
        listetot2=[]
        sousliste=[]
        compteur=0
        for k in range(len(listetot)):
            if k==len(listetot)-1:
                listetot2.append(sousliste)
                compteur+=1
                sousliste=[]
            if listetot[k][3]<ballon[compteur]:
                sousliste.append(listetot[k])
            else:
                listetot2.append(sousliste)
                sousliste=[]
                compteur+=1
        print("nb de features:", len(listetot2))
        listefeatures=[]
        for element in listetot2:
            if len(element)!=0:
                feature_vector = [playerID]
                dist=length(element)
                bc=barycenter(element)
                if "nb_wmax" in Listefeatures:
                    feature_vector.append(nb_wmax(element)) #feature 0
                if "nb_wmin" in Listefeatures:
                    feature_vector.append(nb_wmin(element)) #feature 1
                if "angular_speed_min" in Listefeatures:
                    feature_vector.append(angular_speed_min(element)) #2
                if "angular_speed_max" in Listefeatures:
                    feature_vector.append(angular_speed_max(element)) #3
                if "angular_speed_avg" in Listefeatures:
                    feature_vector.append(angular_speed_avg(element)) #4
                if "dist/diag" in Listefeatures:
                    feature_vector.append(dist/diag)
                if "game area" in Listefeatures:
                    feature_vector.append(np.float64(0.5 + bc[0] / game_area[0])) # between 0 and 1
                    feature_vector.append(np.float64(0.5 + bc[1] / game_area[1]))
                if location_max(element) == 0 and "barycenter distance" in Listefeatures:
                    feature_vector.append(np.float64(0))
                elif "barycenter distance" in Listefeatures:
                    feature_vector.append(location(element)/location_max(element))
                angles = 0.5 + np.array(orientation_feat(element)) / 180 # between 0 and 1
                if "angles" in Listefeatures:
                    feature_vector.append(angles[0][0]) #first orientation of traj
                    feature_vector.append(angles[0][1])
                    feature_vector.append(angles[1][1])#last orientation of traj
                    feature_vector.append(angles[1][1])
                if "nb turns" in Listefeatures:
                    feature_vector.append(nb_turns(element, limit_angle))
                if "velocity average" in Listefeatures:
                    feature_vector.append(velocity_avg(element))
                if "velocity min" in Listefeatures:
                    feature_vector.append(velocity_min(element))
                if "velocity max" in Listefeatures:
                    feature_vector.append(velocity_max(element))
                if "number of mins" in Listefeatures:
                    feature_vector.append(nb_vmin(element))
                if "number of maxs" in Listefeatures:
                    feature_vector.append(nb_vmax(element))
                listefeatures.append(feature_vector)
        return listefeatures
    else:
        listetot=[]
        for k in range(len(traj)):
            listetot+=traj[k]
        if len(listetot)==0:
            return []
        tempslimit=time+listetot[0][3]
        listetot2=[]
        sousliste=[]
        for k in range(len(listetot)):
            if k==len(listetot)-1:
                listetot2.append(sousliste)
                tempslimit=listetot[k][3]+time
                sousliste=[]
            if listetot[k][3]<=tempslimit:
                sousliste.append(listetot[k])
            else:
                listetot2.append(sousliste)
                tempslimit=listetot[k][3]+time
                sousliste=[]
        #print("nb de features:", len(listetot2))
        listefeatures=[]
        for element in listetot2:
            if len(element)!=0:
                feature_vector = [playerID]
                dist=length(element)
                bc=barycenter(element)
                if "nb_wmax" in Listefeatures:
                    feature_vector.append(nb_wmax(element)) #feature 0
                if "nb_wmin" in Listefeatures:
                    feature_vector.append(nb_wmin(element)) #feature 1
                if "angular_speed_min" in Listefeatures:
                    feature_vector.append(angular_speed_min(element)) #2
                if "angular_speed_max" in Listefeatures:
                    feature_vector.append(angular_speed_max(element)) #3
                if "angular_speed_avg" in Listefeatures:
                    feature_vector.append(angular_speed_avg(element)) #4
                if "dist/diag" in Listefeatures:
                    feature_vector.append(dist/diag)
                if "game area" in Listefeatures:
                    feature_vector.append(np.float64(0.5 + bc[0] / game_area[0])) # between 0 and 1
                    feature_vector.append(np.float64(0.5 + bc[1] / game_area[1]))
                if location_max(element) == 0 and "barycenter distance" in Listefeatures:
                    feature_vector.append(np.float64(0))
                elif "barycenter distance" in Listefeatures:
                    feature_vector.append(location(element)/location_max(element))
                angles = 0.5 + np.array(orientation_feat(element)) / 180 # between 0 and 1
                if "angles" in Listefeatures:
                    feature_vector.append(angles[0][0]) #first orientation of traj
                    feature_vector.append(angles[0][1])
                    feature_vector.append(angles[1][1])#last orientation of traj
                    feature_vector.append(angles[1][1])
                if "nb turns" in Listefeatures:
                    feature_vector.append(nb_turns(element, limit_angle))
                if "velocity average" in Listefeatures:
                    feature_vector.append(velocity_avg(element))
                if "velocity min" in Listefeatures:
                    feature_vector.append(velocity_min(element))
                if "velocity max" in Listefeatures:
                    feature_vector.append(velocity_max(element))
                if "number of mins" in Listefeatures:
                    feature_vector.append(nb_vmin(element))
                if "number of maxs" in Listefeatures:
                    feature_vector.append(nb_vmax(element))
                listefeatures.append(feature_vector)
        return listefeatures


def feature_vector_bucket(traj, playerID, game_area, limit_angle=0.25 ,Listefeatures=["nb_wmax", "nb_wmin", "angular_speed_min","angular_speed_max", "angular_speed_avg", "dist/diag","game area","barycenter distance","angles","nb turns","velocity average","velocity min","velocity max","number of mins","number of maxs"] ):
    diag = np.sqrt(game_area[0]**2 + game_area[1]**2)
    
    feature_vector = [playerID]
    
    feature_vector.append(length(traj)/diag)
    
    bc = barycenter(traj)
    feature_vector.append(np.float64(0.5 + bc[0] / game_area[0])) # between 0 and 1
    feature_vector.append(np.float64(0.5 + bc[1] / game_area[1]))
    
    if location_max(traj) == 0:
        feature_vector.append(np.float64(0))
    else:
        feature_vector.append(location(traj)/location_max(traj))
    
    angles = 0.5 + np.array(orientation_feat(traj)) / 180 # between 0 and 1
    feature_vector.append(angles[0]) #first orientation of traj
    feature_vector.append(angles[1]) #last orientation of traj
    
    bucket = bucketize_nb_turns(nb_turns(traj, limit_angle))
    for i in bucket:
        feature_vector.append(i)
    
    v_max = velocity_max(traj)
    if v_max == 0:
        feature_vector.append(0)
        feature_vector.append(0)
        feature_vector.append(0)
    else:
        feature_vector.append(velocity_avg(traj) / v_max)

        feature_vector.append(velocity_min(traj) / v_max)
        feature_vector.append(v_max)
    
    bucket_min = bucketize_nb_v(nb_vmin(traj))
    bucket_max = bucketize_nb_v(nb_vmax(traj))
    for i in bucket_min:
        feature_vector.append(i)
    for j in bucket_max:
        feature_vector.append(j)
    
    return feature_vector


# The function `feature_vectors_game` allows to create the feature vectors over all the trajectories between the gathering of two bubbles of one game. The returned array is an array of multiple 13x5 arrays (the five feature vectors, containing 13 features each, corresponding to the five trajectories of each wave).

import ntpath

def feature_vectors_game(game_file, time, limit_angle=0.25, Listefeatures=["nb_wmax", "nb_wmin", "angular_speed_min","angular_speed_max", "angular_speed_avg", "angular_speed", "dist/diag","game area","barycenter distance","angles","nb turns","velocity average","velocity min","velocity max","number of mins","number of maxs"]):
    balloon=list(create_df_balloon(game_file)['timeOfDestroy'])
    #print(balloon)
    trajectories = sub_trajectories(game_file)
    nb_waves = len(trajectories)
    trajconcac=[]
    game_area=[0,0]
    for traj in trajectories:
        for k in range(len(traj)):
            trajconcac+=traj[k]
    x_list = [tuple_trajectory[0] for tuple_trajectory in  trajconcac]
    x_min = min(x_list)
    x_max = max(x_list)
    y_list = [tuple_trajectory[1] for tuple_trajectory in  trajconcac]
    y_min = min(y_list)
    y_max = max(y_list)
    game_area[0] = (x_max - x_min)
    game_area[1] = (y_max - y_min)
    #print(trajectories[-1][-1][3])
    playerID = ntpath.basename(game_file)[:-5] #gets the name of the file as the player identity
    vectors = []
    for traj1 in trajectories:
        if time=='wave':
            vectors.append(feature_vector(traj1, playerID, balloon, game_area,time,limit_angle, Listefeatures))
        else:
            for feature_vec in feature_vector(traj1, playerID, balloon, game_area,time,limit_angle, Listefeatures):
                vectors.append(feature_vec)

    return np.array(vectors)

#Time has three possible values: ('wave', 'pop' or float),
#- wave: The features will be computed in a whole wave (sequence of 5 balloons)
#- pop: The features will be computed between two consecutive balloons
#- float: The features will be computed in time intervals of time = float. You must put a float as argument. If you choose 1, for instance, the features will be computed at each second


def simple_features_generator(game_list,playerno, time='wave', limit_angle=0.25, Listefeatures=["nb_wmax", "nb_wmin", "angular_speed_min","angular_speed_max", "angular_speed_avg", "angular_speed", "dist/diag","game area","barycenter distance","angles","nb turns","velocity average","velocity min","velocity max","number of mins","number of maxs"]):
    features=[]
    labels=[]
    for file in game_list:
        for layer1 in feature_vectors_game(file,time, limit_angle, Listefeatures):
            if len(layer1)==0:
                pass
            else:
                labels.append(layer1[0])
                features.append(layer1[1:])
    np.savetxt('features{}{}.csv'.format(time,playerno), features, delimiter=",", fmt='%s')
    np.savetxt('output{}{}.csv'.format(time,playerno), labels,delimiter=",", fmt='%s')
    return features, labels


# The following functions provide different shapes for the feature vector. This way of creating the feature vector could be improved by using tensorflow and its feature vectors, instead of creating it "by hand".
# * "concat" means all features are concatenated into one numpy vector for each sample
# * "bucket" means it uses the bucketized version of the feature vector (for nb_turns, nb_vmin, nb_vmax)
# * "hands"  means it uses the hand used to play as label instead of the player's ID


def feature_vectors_game_concat(game_file, game_area = [21,10]):
    trajectories = np.array(sub_trajectories(game_file))
    nb_waves = len(trajectories)
    playerID = 1 #int(parse_root(game_file)[2][0].text) ## CHANGE THIS PART #################################
    vectors = []
    
    for i in range(nb_waves):
        vectors.append([])
        for traj in trajectories[i]:
            vectors[i] = vectors[i] + list(feature_vector(traj, playerID, game_area)[1:])
        vectors[i].append(playerID) ## CHANGE THIS PART #################################
    
    return np.array(vectors)

def feature_vectors_bucket_game_concat(game_file, game_area = [21,10]):
    trajectories = np.array(sub_trajectories(game_file))
    nb_waves = len(trajectories)
    playerID = 1#int(parse_root(game_file)[2][0].text)
    vectors = []
    
    for i in range(nb_waves):
        vectors.append([])
        for traj in trajectories[i]:
            vectors[i] = vectors[i] + list(feature_vector_bucket(traj, playerID, game_area)[1:])
        vectors[i].append(playerID)
    
    return np.array(vectors)

def feature_vectors_bucket_game_concat_hands(game_file, game_area = [21,10]):
    trajectories = np.array(sub_trajectories(game_file))
    nb_waves = len(trajectories)
    if parse_root(game_file)[2][2].text == 'false':
        useRightHand = 0
    else:
        useRightHand = 1
    vectors = []
    
    for i in range(nb_waves):
        vectors.append([])
        for traj in trajectories[i]:
            vectors[i] = vectors[i] + list(feature_vector_bucket(traj, useRightHand, game_area)[1:])
        vectors[i].append(useRightHand)
    
    return np.array(vectors)


# Finally we provide a function to get the agregation of all feature vectors over multiple game files, where *game_files* is the list of the names (String type) of all the game files to be considered.



def agregate_feature_vectors(game_files):
    vectors = []
    for file in game_files:
        vectors = vectors + list(feature_vectors_game_concat(file))
    
    return np.array(vectors)




def agregate_feature_vectors_bucket(game_files):
    vectors = []
    for file in game_files:
        vectors = vectors + list(feature_vectors_bucket_game_concat(file))
    
    return np.array(vectors)

def agregate_feature_vectors_bucket_hands(game_files):
    vectors = []
    for file in game_files:
        vectors = vectors + list(feature_vectors_bucket_game_concat_hands(file))
    
    return np.array(vectors)


# # Export of the final data

def export_feature_vectors(vectors, name):
    np.savetxt(name, vectors, delimiter=",")

#Correlation Matrix Creation

for i in players_to_study:
	simple_features_generator([game_files[i-1]],i)
	simple_features_generator([game_files[i-1]],i, time='pop')


for i in players_to_study:
	X = np.genfromtxt('./featureswave{}.csv'.format(i), delimiter=',')
	y = np.genfromtxt('./outputwave{}.csv'.format(i), delimiter=',')

	plt.figure()
	sns_plot = sns.heatmap(pd.DataFrame(X).corr(), cmap='seismic')
	plt.savefig('correlation_p{}.png'.format(i), dpi=400)  
	plt.clf()

	# In[85]:

	plt.figure()
	scatter_matrix = pd.plotting.scatter_matrix(pd.DataFrame(X), c='darkorchid', figsize=(15, 15), marker='o',hist_kwds={'bins': 20, 'color' : 'darkmagenta'}, s=10, alpha=.6)
	plt.savefig(r"scatter_matrix_p{}.png".format(i) )
	plt.clf()



#K Means Classification 

X=[]
y=[]

for i in players_to_study:
    for array in np.genfromtxt('featureswave{}.csv'.format(i), delimiter=','):
        X.append(array)
    for array in np.genfromtxt('outputwave{}.csv'.format(i), delimiter=','):
        y.append(array)

X = np.asarray(X)
y= np.asarray(y)


model_pca3 = PCA(n_components=3)

# On entraîne notre modèle (fit) sur les données
model_pca3.fit(X)

# On applique le résultat sur nos données :
X_reduced3 = model_pca3.transform(X)


# In[89]:


# Création de la figure 3D
fig = plt.figure(0)
ax = Axes3D(fig, elev=-120, azim=50)

# Affichage des valeurs
ax.scatter(X_reduced3[:, 0], X_reduced3[:, 1], X_reduced3[:, 2], c=y, cmap='Spectral')


# Let's plot the 2-dimension graphs to have a better ideia on how the data looks like

# In[90]:


#plt.figure(1)
##########Real
plt.subplot(1, 2, 1)
plt.scatter(X_reduced3[:, 0], X_reduced3[:, 1], c=y, cmap='Set1', s = 3)
plt.xlabel('Dimension 0')
plt.ylabel('Dimension 1')
plt.title('Real Output')

plt.subplot(1, 2, 2)
plt.scatter(X_reduced3[:, 0], X_reduced3[:, 2], c=y, cmap='Set1', s = 3)
plt.xlabel('Dimension 0')
plt.ylabel('Dimension 2')
plt.title('Predicted by 2-PCA Output')


# 2-Dimension PCA

# In[91]:


model_pca2 = PCA(n_components=2)

# On entraîne notre modèle (fit) sur les données
model_pca2.fit(X)

# On applique le résultat sur nos données :
X_reduced2 = model_pca2.transform(X)


# In[92]:


plt.scatter(X_reduced2[:, 0], X_reduced2[:, 1], c=y, cmap='Set1', s = 3)
plt.xlabel('Dimension 0')
plt.ylabel('Dimension 1')
plt.title('Real Output')


# Applying K-Means Algorithm

# For the 3d-PCA

# Let's start with the data generated from 2 players (8 and 9). So n_clusters = 3


n_clusters = 2

model3=KMeans(n_clusters)
model3.fit(X_reduced3)
pred_pca3 = model3.predict(X_reduced3)



plt.subplot(1, 2, 1)
plt.scatter(X_reduced3[:, 0], X_reduced3[:, 1], c=y, cmap='Set1', s = 3)
plt.xlabel('Dimension 0')
plt.ylabel('Dimension 1')
plt.title('Real Output')

plt.subplot(1, 2, 2)
plt.scatter(X_reduced3[:, 0], X_reduced3[:, 1], c=pred_pca3, cmap='Set1', s = 3)
plt.xlabel('Dimension 0')
plt.ylabel('Dimension 1')
plt.title('Predicted by 2-PCA Output')

plt.subplot(1, 2, 1)
plt.scatter(X_reduced3[:, 0], X_reduced3[:, 2], c=y, cmap='Set1')
plt.xlabel('Dimension 0')
plt.ylabel('Dimension 2')
plt.title('Real Output')

#plt.figure(2)
plt.subplot(1, 2, 2)
plt.scatter(X_reduced3[:, 0], X_reduced3[:, 2], c=pred_pca3, cmap='Set1')
plt.xlabel('Dimension 0')
plt.ylabel('Dimension 2')
plt.title('Predicted by 3-PCA Output')


# Let's try now for the 2d-PCA


model2=KMeans(n_clusters)
model2.fit(X_reduced2)
pred_pca2 = model2.predict(X_reduced2)

plt.subplot(1, 2, 1)
plt.scatter(X_reduced3[:, 0], X_reduced3[:, 1], c=y, cmap='Set1', s = 10)
plt.xlabel('Dimension 0')
plt.ylabel('Dimension 1')
plt.title('Real Output')

plt.subplot(1, 2, 2)
plt.scatter(X_reduced3[:, 0], X_reduced3[:, 1], c=pred_pca2, cmap='Set1', s = 10)
plt.xlabel('Dimension 0')
plt.ylabel('Dimension 1')
plt.title('Predicted by 2-PCA Output')


# Not that bad!!
#Start KNN Analysis

X=[]
y=[]

for i in players_to_study:
    for array in np.genfromtxt('featureswave{}.csv'.format(i), delimiter=','):
        X.append(array)
    for array in np.genfromtxt('outputwave{}.csv'.format(i), delimiter=','):
        y.append(array)

X = np.asarray(X)
y= np.asarray(y)


## Applying PCA

model_pca3 = PCA(n_components=3)

# On entraîne notre modèle (fit) sur les données
model_pca3.fit(X)

# On applique le résultat sur nos données :
X_reduced3 = model_pca3.transform(X)

# On crée notre modèle pour obtenir 2 composantes
model_pca2 = PCA(n_components = 2)

# On entraîne notre modèle (fit) sur les données
model_pca2.fit(X)

# On applique le résultat sur nos données :
X_reduced2 = model_pca2.transform(X)


# ## Applying K-means    
# 

# In[19]:


def visualiser_modele(model):
    
    #levels = [0, 1, 2, 3]
    levels = [7, 8, 9, 10]
    colors = ['red', 'yellow', 'blue']
    cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors)
    
    # On crée un tableau de coordonnées pour chaque point du plan (une grille)
    xx, yy = np.meshgrid(np.arange(-4, 4, 0.1), np.arange(-1.5, 1.6, 0.1))
    X_grid = np.c_[xx.flatten(), yy.flatten()]

    # On calcule ce que prédit le classifier en chaque point de ce plan
    y_grid = model.predict(X_grid)

    # On dessine ce que le modèle prévoit sur le plan
    plt.contourf(xx, yy, y_grid.reshape(xx.shape), cmap='Spectral')

    # On affiche les points du training en contour noir
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker = "o", norm=norm, cmap=cmap, edgeColor='black')

    # On affiche les points du test en contour blanc
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker ="o",norm=norm, cmap=cmap, edgeColor='white', linewidths=2)
    
#applying K-means    
# Mélanger le dataset (penser à mélanger les features et les labels de la même façon...)
p = np.random.permutation(len(X))
X, y = X[p], y[p]

# Réduire à 2 dimensions
model_pca2.fit(X)
X_reduced2 = model_pca2.transform(X)


# Séparer training et test
training_ratio = 0.6
l = len(X)
X_train = X_reduced2[:int(l*training_ratio)]
X_test = X_reduced2[int(l*training_ratio):]
y_train = y[:int(l*training_ratio)]
y_test = y[int(l*training_ratio):]


# Créer le classifier
model_knn = KNeighborsClassifier(3)

# Entrainer le classifier sur les données d'entrainement (X_train et y_train)
model_knn.fit(X_train, y_train)

# Evaluer
y_pred = model_knn.predict(X_test)

true_positive = sum(y_pred == y_test)
n_predictions = len(y_test)
accuracy = true_positive / n_predictions

print("Précision KNN=3, PCA=2: %i%%" % (accuracy * 100))

# Afficher les résultats
visualiser_modele(model_knn)
# Let's try a 3-dimension PCA now

# In[108]:


p = np.random.permutation(len(X))
X, y = X[p], y[p]

model_pca3 = PCA(n_components=3)
model_pca3.fit(X)
X_reduced3 = model_pca3.transform(X)

# Séparer training et test
training_ratio = 0.6
l = len(X)
X_train = X_reduced3[:int(l*training_ratio)]
X_test = X_reduced3[int(l*training_ratio):]
y_train = y[:int(l*training_ratio)]
y_test = y[int(l*training_ratio):]


# Créer le classifier
model_knn = KNeighborsClassifier(3)

# Entrainer le classifier sur les données d'entrainement (X_train et y_train)
model_knn.fit(X_train, y_train)

# Evaluer
y_pred = model_knn.predict(X_test)

true_positive = sum(y_pred == y_test)
n_predictions = len(y_test)
accuracy = true_positive / n_predictions

print("Précision KNN =3, PCA = 3: %i%%" % (accuracy * 100))


# The precision increased from 67% to 60% using a PCA of dimension 3. Let's analyse using a PCA of dimension 10

# In[112]:


model_pca10 = PCA(n_components=8)
model_pca10.fit(X)
X_reduced10 = model_pca3.transform(X)

# Séparer training et test
training_ratio = 0.6
l = len(X)
X_train = X_reduced10[:int(l*training_ratio)]
X_test = X_reduced10[int(l*training_ratio):]
y_train = y[:int(l*training_ratio)]
y_test = y[int(l*training_ratio):]


# Créer le classifier
model_knn = KNeighborsClassifier(3)

# Entrainer le classifier sur les données d'entrainement (X_train et y_train)
model_knn.fit(X_train, y_train)

# Evaluer
y_pred = model_knn.predict(X_test)

true_positive = sum(y_pred == y_test)
n_predictions = len(y_test)
accuracy = true_positive / n_predictions

print("Précision KNN =3, PCA =8 : %i%%" % (accuracy * 100))



#Start analysis for each pop
X=[]
y=[]

for i in players_to_study:
    for array in np.genfromtxt('featurespop{}.csv'.format(i), delimiter=','):
        X.append(array)
    for array in np.genfromtxt('outputpop{}.csv'.format(i), delimiter=','):
        y.append(array)

X = np.asarray(X)
y= np.asarray(y)


#applying K-Neighbors    
# Mélanger le dataset (penser à mélanger les features et les labels de la même façon...)
p = np.random.permutation(len(X))
X, y = X[p], y[p]

# Réduire à 2 dimensions
model_pca2.fit(X)
X_reduced2 = model_pca2.transform(X)

# Séparer training et test
training_ratio = 0.6
l = len(X)
X_train = X_reduced2[:int(l*training_ratio)]
X_test = X_reduced2[int(l*training_ratio):]
y_train = y[:int(l*training_ratio)]
y_test = y[int(l*training_ratio):]


# Créer le classifier
model_knn = KNeighborsClassifier(3)

# Entrainer le classifier sur les données d'entrainement (X_train et y_train)
model_knn.fit(X_train, y_train)

# Evaluer
y_pred = model_knn.predict(X_test)

true_positive = sum(y_pred == y_test)
n_predictions = len(y_test)
accuracy = true_positive / n_predictions

print("Précision Pop KNN2 PCA2: %i%%" % (accuracy * 100))

# Afficher les résultats
visualiser_modele(model_knn)


# It seems that the precision increases when compared to the data sampled with 1Hz. This can be explained by less "noise" in the data

# Let's try now with a 10 dimension PCA

# In[132]:


p = np.random.permutation(len(X))
X, y = X[p], y[p]

model_pca10 = PCA(n_components=10)
model_pca10.fit(X)
X_reduced10 = model_pca10.transform(X)

# Séparer training et test
training_ratio = 0.6
l = len(X)
X_train = X_reduced10[:int(l*training_ratio)]
X_test = X_reduced10[int(l*training_ratio):]
y_train = y[:int(l*training_ratio)]
y_test = y[int(l*training_ratio):]


# Créer le classifier
model_knn = KNeighborsClassifier(3)

# Entrainer le classifier sur les données d'entrainement (X_train et y_train)
model_knn.fit(X_train, y_train)

# Evaluer
y_pred = model_knn.predict(X_test)

true_positive = sum(y_pred == y_test)
n_predictions = len(y_test)
accuracy = true_positive / n_predictions

print("Précision KNN3 PCA10: %i%%" % (accuracy * 100))


# ## With 10 dimension PCA we notice that our accuracy is even worse (68% against the 75% we had before). This is mainly due to the fact that we have too little amount of data, and at each time we run the algorithm, we get a very different precision (ranging from 45 to 75%). We are not able to conclude anything from this study, given that randomly choosing the players we could get a precision of 50%
