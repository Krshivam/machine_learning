import sys
import numpy as np
import cv2
from sklearn.metrics import mean_squared_error


def get_pixel():
    x=[]
    s=[]
    l = {0:'/root/Pictures/0.jpeg',2:'/root/Pictures/2.jpeg',3:'/root/Pictures/3.jpeg',
    4:'/root/Pictures/4.jpeg',5:'/root/Pictures/5.jpeg',6:'/root/Pictures/6.jpeg',
    7:'/root/Pictures/7.jpeg',8:'/root/Pictures/8.jpeg',9:'/root/Pictures/9.jpeg',
    10:'/root/Pictures/10.jpeg',13:'/root/Pictures/13.jpeg',14:'/root/Pictures/14.jpeg',
    15:'/root/Pictures/15.jpeg',16:'/root/Pictures/16.jpeg',18:'/root/Pictures/18.jpeg',
    20:'/root/Pictures/20.jpeg'}
    for key in l:
        a=[]
    	img=cv2.imread(l[key])
    
        blue = img[100,100,0]
        a.append(blue)
        a.append(key)
        x.append(a)
    return x




def relu(f):
    if f<0:
        return 0
    else:
        return f

def p_w_n():
    
    input_data = np.array(get_pixel())
    weights = {'node_0_0':[1,1],'node_1_0':[0,1],'node_0_1':[-1,1],'node_1_1':[2,3],'output':[1,2]}
    for i in range(len(input_data)):

        
        node_0_input_0= (input_data[i]*np.array(weights['node_0_0'])).sum()
        node_0_output_0 = relu(node_0_input_0)
        node_0_input_1 = (input_data[i]*np.array(weights['node_1_0'])).sum()
        node_0_output_1 = relu(node_0_input_1)
        layer_1_input = np.array([node_0_output_0,node_0_output_1])
        layer_1_input_0 = (layer_1_input[0]*np.array(weights['node_0_1'])).sum()
        layer_1_output_0 = relu(layer_1_input_0)
        layer_1_input_1 =  (layer_1_input[1]*np.array(weights['node_1_1'])).sum()
        layer_1_output_1 = relu(layer_1_input_1)
        layer_2_output = np.array([layer_1_output_0,layer_1_output_1])
        final_output = (layer_2_output*np.array(weights['output'])).sum()
        
        print final_output




p_w_n()      
    





