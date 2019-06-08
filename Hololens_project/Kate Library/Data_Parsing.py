#!/usr/bin/env python
# coding: utf-8

# # How to use this notebook ?
# * In the last cell, enter the json files of data you wish to use in the game_files list.
# * In relative_path, put the path of the folder containing Hololens data (dafault is './Hololens_data/')
# * Call the function simple_features_generator with the following parameters:
# - 'game_list': A list with the path of the files to be considered in the features.csv file
# - 'limit_angle': the angle for wich we consider a change in the velocity (nb_v)
# - 'ListeFeatures': A list with the attributes our parser will generate

# In[1]:


import json
import numpy as np
import pandas as pd


# In[4]:


def norm(vect):
    sum = 0
    
    for el in vect:
        sum += el**2
    
    return np.sqrt(sum)


# Let's create a Panda's dataFrame with position, time, rotatio, BPM  of each frame of the game and a second Dataframe with the balloons gathering data

# In[40]:


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
    
    #Drop all the columns that are already included in head_positions column as a tuple
    df_game.drop(['x','y','z', 'headRotationY', 'headPos'], axis = 1,inplace = True)               
    
    
    return df_game


# In[41]:


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


# * The function `hand_positions` extracts the positions of the head/hand of the user along with the time corresponding to those positions. It returns an array of shape [(x, y, z, t, rotation)] (length number_of_position, with 5 elements arrays representing (x, y, z, t, rotation)). We won't distinguish between head/hand in the code.

# In[47]:


def hand_positions(game_file):
    return list(create_df_hand(game_file)['head_positions'])
    


# * The function `bubble_pop` extracts the time of each game event corresponding to the pop of a bubble by the player. It returns an array of shape [t] (length number_of_bubble_poped).

# In[48]:


def bubble_pop(game_file):
    return list(create_df_balloon(game_file)['timeOfDestroy'])


# # Extraction of sub-trajectories & features
# The function `sub_trajectories` returns an array of shape [[*[(x,y,z,t,rotation),(x,y,z,t,rotation),...]*, for each bubble in wave], for each wave]. To access all positions and time of the trajectory between the *i* and *i+1* bubble of the *n* wave : *sub_trajectories[n-1][i]*.

# In[51]:


def sub_trajectories(game_file):
    hand_position = hand_positions(game_file)
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


# We define some functions to extract interesting features from trajectories. We first look for Static features : 
# * `length` returns the length of the trajectory *traj*
# * `barycenter` returns the barycenter of the trajectory *traj* in shape (x,y,z)
# * `location` returns the average distance of each point to the barycenter of the trajectory *traj*
# * `location_max` returns the maximum distance between a point of the trajectory and the barycenter of this trajectory
# * `orientation` returns the angle between points the line between *(x1, y1, z1)* and *(x2, y2, z2)* and the horizontal axis (in degrees)
# * `orientation_feat` returns the preceeding feature for the first two points and the last two points of the trajectory *traj*
# * `nb_turns` returns the number of turns in the trajectory *traj*, where a turn is detected if the orientation between two consecutive couples of points varies of more than *limit_angle*

# In[55]:


def length(traj):
    l = 0
    
    for i in range(len(traj)-1):
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
        loc_avg += np.sqrt((traj[i][0] - p[0])**2 + (traj[i][1] - p[1])**2+(traj[i][2]-p[2])**2)
        
    return loc_avg/n

def location_max(traj):
    n = len(traj)
    p = barycenter(traj)
    if n>0:
        l_max = np.max([np.sqrt((traj[i][0] - p[0])**2 + (traj[i][1] - p[1])**2+(traj[i][2]-p[2])**2) for i in range(n)])
        return l_max
    else:
        return 0

    
def orientation(x1, x2 , y1, y2, z1, z2):
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
    
    return nb_turns


# We then define dynamic features:
# * `velocity` returns the list of the point to point velocities over the whole trajectory *traj*
# * `angular speed` returns the list of the point to point angular speed over the whole trajectory *traj*
# * `velocity_avg` returns the average velocity over the trajectory *traj*
# * `angular_speed_avg` returns the average angular speed over the trajectory *traj*
# * `velocity_max` returns the greatest velocity over the trajectory *traj*
# * `angular_speed_max` returns the greatest angular speed over the trajectory *traj*
# * `velocity_min` returns the lowest velocity over the trajectory *traj*
# * `angular_speed_min` returns the lowest angular speed over the trajectory *traj*
# * `nb_vmin` returns the number of local minimum of velocity
# * `nb_wmin` returns the number of local minimum of angular speed
# * `nb_vmax` returns the number of local maximum of velocity
# * `nb_wmax` returns the number of local maximum of angular speed

# In[62]:


def velocity(traj):
    velocity = []
    
    for i in range(len(traj) - 1):
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


# 

# In[67]:


def feature_vector(traj, playerID, ballon, game_area, time, limit_angle=0.25, Listefeatures=["nb_wmax", "nb_wmin", "angular_speed_min","angular_speed_max", "angular_speed_avg", "dist/diag","game area","barycenter distance","angles","nb turns","velocity average","velocity min","velocity max","number of mins","number of maxs"]):    
    '''
    The function `feature_vector` extracts features from the trajectory.
    'traj': [(x,y,z,t,rotation)]*
    'game_area' : [x,y] vector with the dimensios of to be used to normalize positions.
    'Time' has three possible values: ('wave', 'pop' or float),
        #- wave: The features will be computed in a whole wave (sequence of 5 balloons)
        #- pop: The features will be computed between two consecutive balloons
        #- float: The features will be computed in time intervals of time = float. You must put a float as argument. If you choose 1, for instance, the features will be computed at each second
    'limit_angle': the angle for wich we consider a change in the velocity (nb_v)
    'ListeFeatures': A list with the attributes our parser will generate
    
    '''
    
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
    


# The function `feature_vectors_game` allows to create the feature vectors over all the trajectories between the gathering of two bubbles of one game. The returned array is an array of multiple 13x5 arrays (the five feature vectors, containing 13 features each, corresponding to the five trajectories of each wave).

# In[58]:


import ntpath

def feature_vectors_game(game_file, time, limit_angle=0.25, Listefeatures=["nb_wmax", "nb_wmin", "angular_speed_min","angular_speed_max", "angular_speed_avg", "angular_speed", "dist/diag","game area","barycenter distance","angles","nb turns","velocity average","velocity min","velocity max","number of mins","number of maxs"]):
    '''
    The function `feature_vectors_game` allows to create the feature vectors over all the trajectories between the gathering of two bubbles of one game. The returned array is an array of multiple 13x5 arrays (the five feature vectors, containing 13 features each, corresponding to the five trajectories of each wave).
    'game_file': the path of where the json file is located
    'Time' has three possible values: ('wave', 'pop' or float),
        #- wave: The features will be computed in a whole wave (sequence of 5 balloons)
        #- pop: The features will be computed between two consecutive balloons
        #- float: The features will be computed in time intervals of time = float. You must put a float as argument. If you choose 1, for instance, the features will be computed at each second
    'limit_angle': the angle for wich we consider a change in the velocity (nb_v)
    'ListeFeatures': A list with the attributes our parser will generate
    
    '''
    
    balloon=list(create_df_balloon(game_file)['timeOfDestroy'])
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
    playerID = ntpath.basename(game_file)[:-5] #gets the name of the file as the player identity
    vectors = []
    for traj1 in trajectories:
        if time=='wave':
            vectors.append(feature_vector(traj1, playerID, balloon, game_area,time,limit_angle, Listefeatures))
        else:
            for feature_vec in feature_vector(traj1, playerID, balloon, game_area,time,limit_angle, Listefeatures):
                vectors.append(feature_vec)

    return np.array(vectors)


# simple_features_generator Allows to create two different csv files: features.csv with all the features computed by features_vector_game and output.csv, which is a file with the identity of the players, which in our case will be the name of the json file (each player has a different csv file)

# In[59]:


def simple_features_generator(game_list,time='wave', limit_angle=0.25, Listefeatures=["nb_wmax", "nb_wmin", "angular_speed_min","angular_speed_max", "angular_speed_avg", "angular_speed", "dist/diag","game area","barycenter distance","angles","nb turns","velocity average","velocity min","velocity max","number of mins","number of maxs"]):
    '''
    Parameters:
    'game_list': A list with the path of the files to be considered in the features.csv file
    'Time' has three possible values: ('wave', 'pop' or float),
        #- wave: The features will be computed in a whole wave (sequence of 5 balloons)
        #- pop: The features will be computed between two consecutive balloons
        #- float: The features will be computed in time intervals of time = float. You must put a float as argument. If you
    'limit_angle': the angle for wich we consider a change in the velocity (nb_v)
    'ListeFeatures': A list with the attributes our parser will generate

    Return:
    The function returns a tuple (X, y) where X is an array of features (each line is an example) and y is a list of labels.
    It also saves at the current folder two csv files: features.csv and output.csv with X and y respectively

    '''
    features=[]
    labels=[]
    for file in game_list:
        for layer1 in feature_vectors_game(file,time, limit_angle, Listefeatures):
            if len(layer1)==0:
                pass
            else:
                labels.append(layer1[0])
                features.append(layer1[1:])
    np.savetxt('features.csv', features, delimiter=",", fmt='%s')
    np.savetxt('output.csv', labels,delimiter=",", fmt='%s')
    return features, labels

# # Export of the final data

# In[71]:



#standart values to relative_path and game_files
relative_path = './Hololens_data/'

game_files=[relative_path + '1.json',
           relative_path + '3.json',
           relative_path + '4.json',
           relative_path + '5.json',
           relative_path + '6.json',
           relative_path + '7.json',
           relative_path + '8.json',
           relative_path + '9.json',
           relative_path + '10.json'] #here we add all the files we want to train the algorithm with


# In[73]:


game_files=[
           relative_path + '5.json',
           relative_path + '6.json',
           relative_path + '7.json',
           relative_path + '8.json',
           relative_path + '9.json',
           relative_path + '10.json']


# In[74]:


# example of how to generate a file
# X, y = simple_features_generator(game_files, 1)

