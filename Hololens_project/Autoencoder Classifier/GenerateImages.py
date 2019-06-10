
# coding: utf-8

# In[48]:


import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import os

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
    
    #Here we create a column with a 4-element tuple: (x,y,z,t) for each dataframe
    df_game['hand_positions'] = df_game[['x', 'y', 'z', 'timeStamp']].apply(lambda x: tuple(x), axis=1)
    
    return df_game

def create_df_balloon(game_file):
        
    with open(game_file) as json_file:
        data = json.load(json_file)
        
        
    df_balloon = pd.DataFrame(data['datasList'][0]['listLevelDatas'][0]['listBalloonDatas'])
    for i in range(1,len(data['datasList'][0]['listLevelDatas'])):
        df_balloon = pd.concat([df_balloon, pd.DataFrame(data['datasList'][0]['listLevelDatas'][i]['listBalloonDatas'])])
    
    return df_balloon


def hand_positions(game_file):
    return list(create_df_hand(game_file)['hand_positions'])


def balloon_positions(game_file):
    return list(create_df_balloon(game_file)['balloonInitialPosition'])

def break_time(t): #Spilts the entire time vector into multiple sub vectors when a new wave is started 
    time_sets={}
    sub_t=[]
    k=0
    for i in range(len(t)-1):
        if t[i+1]>t[i]:
            sub_t.append(t)
        else:
            time_sets[k]=sub_t
            k+=1
    return (time_sets,k)

def derivative(v,t):
    time_sets,k= break_time(t)
    velocity = [0]*len(v)
    for i in range(len(v)):
        try:
            velocity[i] = (v[i+1] -v[i])/(0.06) #0.06 is the average sampling period of the accusation software
        except:
            velocity[i]=0 
            
    for i in range(k): #To Remove discrepancies in the calculations of the derivative
        velocity.remove(max(velocity, key = abs))
    return (velocity)


def timings_balloons(game_file):
    return (np.asarray(create_df_balloon(game_file)['timeOfSpawn']),list(create_df_balloon(game_file)['timeOfDestroy']))

def sampling_times(game_file):
    return (np.asarray(create_df_hand(game_file)['timeStamp']))

def find_increasing(seq):
    found=[]
    for i in range(0,len(seq)-1):
        if abs(seq[i]-seq[i+1]) == 1:
            found.append(seq[i])
        else:
            found.append(seq[i])
            break
    return (found) 

def create_handvector(hand_pos):
    hx=[]
    hy=[]
    hz=[]
    ht=[]
    for i in range(len(hand_pos)):
        hx.append(hand_pos[i][0])
        hy.append(hand_pos[i][1])
        hz.append(hand_pos[i][2])
        ht.append(hand_pos[i][3])
    return (hx,hy,hz,ht)

NumberDatabase = 11


def flip_rotate_image(image):
    image = np.asarray(image)
    rotate_90 = np.rot90(image)
    rotate_180 = np.rot90(rotate_90)
    rotate_270 = np.rot90(rotate_180)
    trans_0 = image.transpose()
    trans_rotate_90 = np.rot90(trans_0)
    trans_rotate_180 = np.rot90(trans_rotate_90)
    trans_rotate_270 = np.rot90(trans_rotate_180)
    return (rotate_90,rotate_180, rotate_270, trans_0, trans_rotate_90, trans_rotate_180, trans_rotate_270)

for database in range(NumberDatabase):
    game_file = 'data' + str(database+1) + '.json' 
    [timeOfSpawn, timeOfDestroy] = timings_balloons(game_file)
    hand_times = sampling_times(game_file)
    N = 12
    (hx,hy,hz,ht) = create_handvector(hand_positions(game_file))
    Loc_X = hx.copy()
    Loc_Z = hz.copy()
    TGlobal = hand_times
    print (database)
    fig = plt.figure(figsize=(1,1)) 
    for i in range(len(timeOfDestroy)):
            UpB = timeOfDestroy[i]
            LowB = timeOfSpawn[i]
            if UpB < LowB:
                LowB=0
            seq = find_increasing(list(np.where((TGlobal>LowB) & (TGlobal<UpB)))[0][:])
            if len(seq)>1:
                TGlobal = np.delete(TGlobal,seq)
                Loc_X = np.delete(Loc_X,seq)
                Loc_Z = np.delete(Loc_Z,seq)
                
                sub_hx = hx[seq[0]: seq[-1]]
                sub_hx = [x-min(sub_hx) for x in sub_hx]
                sub_hz = hz[seq[0]: seq[-1]]
                sub_hz = [z-min(sub_hz) for z in sub_hz]
                """
                plt.scatter(hx[seq[0]: seq[-1]],hz[seq[0]: seq[-1]],c='b')
                plt.savefig('image{}.png'.format(i+1), dpi = N^2)
                plt.clf()
                print (i)
                """
                image_mat = 1000*np.ones((N,N))
                max_x = max([abs(x) for x in sub_hx])
                max_z = max([abs(z) for z in sub_hz])
                delx = (1.1*max_x)/N
                delz = (1.1*max_z)/N
                #print (database)
                try:
                    for k in range(len(sub_hz)):
                        X = int(sub_hx[k]//delx)
                        Z = int(sub_hz[k]//delz)
                        image_mat[N - Z - 1][X]-=1
                    #print(X,Z)
                except:
                    print (max_x,max_z)
                plt.imshow(image_mat, cmap = 'binary')
                plt.axis('off')
                #plt.colorbar()
                plt.savefig('data{}imagebw{}.png'.format(database+1,i+1))
                plt.clf()
                return_images = flip_rotate_image(image_mat)
                for j in range(len(return_images)):
                    plt.axis('off')
                    plt.imshow(return_images[j], cmap = 'binary')
                    plt.savefig('data{}image{}modified{}.png'.format(database+1, i+1,j+1))
                    plt.clf()

