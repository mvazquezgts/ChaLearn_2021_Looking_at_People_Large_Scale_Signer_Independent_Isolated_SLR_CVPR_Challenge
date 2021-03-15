import argparse
import pickle
import os

import numpy as np
from tqdm import tqdm

import pandas as pd

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--label_folder',
                        default='data/autsl',
                        help='the work folder for storing results')
    
    parser.add_argument('--set',
                        required=True,
                        choices={'val', 'test'},
                        help='the work folder for storing results')
   
    parser.add_argument('--inputs',
                        required=True,
                        nargs=2,                  ## set how many results will be emsembled
                        help='Directory containing "epoch1_test_score.pkl" for bone eval/test results')
    
    parser.add_argument('--out',
                        default=None,
                        help='out folder combination')
    
    parser.add_argument('--csv',
                        action='store_true',
                        default=False,
                        help='Generate or not predictions.csv')
    

    arg = parser.parse_args()

    label_folder = arg.label_folder
    set = arg.set
    
    print (label_folder)
    print (set)
    
    inputs = arg.inputs
    print (arg.inputs)
    print (inputs[0])
    print (inputs[1])

    with open('../' + label_folder + '/'+set+'_label.pkl', 'rb') as label:
        label = np.array(pickle.load(label))
        
    arr_predictions = []
    arr_top1 = []
    right_num = total_num = right_num_5 = 0
    arr_out = np.zeros((len(label[0]),226))
    
    for input in inputs:
        with open(os.path.join(input, 'epoch1_test_score.pkl'), 'rb') as r_data:
            r_data = list(pickle.load(r_data).items())
                
        for i in tqdm(range(len(label[0]))):
            arr_out[i]+=r_data[i][1]
            
    
    for i in tqdm(range(len(label[0]))):
        _, l = label[:, i]
        rank_5 = arr_out[i].argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(arr_out[i])
        right_num += int(r == int(l))
        arr_top1.append(int(r == int(l)))
        total_num += 1
        arr_predictions.append(r)
    

    acc = right_num / total_num
    acc5 = right_num_5 / total_num
        
    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))
    
    if (arg.out != None):
        
        with open(os.path.join(input, 'epoch1_test_score.pkl'), 'rb') as f:
            data = pickle.load(f)
        
        for idx, key in enumerate(list(data.keys())):
            data[key] = arr_out[idx]
        
        with open('{}ensemble_score.pkl'.format(arg.out), 'wb') as f:
            pickle.dump(data, f)
            
        print ("Save out pickle: {}ensemble_score.pkl".format(arg.out))  
        
        
        results = np.vstack((list(data.keys()), arr_top1))
        
        results = results.transpose()
        pd.DataFrame(results).to_csv('{}arr_top1.csv'.format(arg.out),index=False,header=False)
        print ('Save out csv: {}arr_top1.csv'.format(arg.out))  
        
    
    if (arg.csv == True):
        
        print ("Generating predictions.csv file")
        with open(os.path.join(input, 'epoch1_test_score.pkl'), 'rb') as f:
            data = pickle.load(f)

        arr_names = []
        for name in list(data.keys()):
            arr_names.append(name[:-11])
            
        results = np.vstack((arr_names, arr_predictions))
        results = results.transpose()
        pd.DataFrame(results).to_csv("predictions.csv",index=False,header=False)
        
    
    print (arr_out.shape)
    print (len(arr_predictions))
    
