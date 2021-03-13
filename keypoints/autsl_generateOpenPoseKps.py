import os
from os import listdir
from os.path import isfile, join
import shutil
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--set', type=str)
parser.add_argument('--split', type=int)
parser.add_argument('--gpu', type=str)
args = parser.parse_args()

def exec_openpose(vid, gpu='0'):  
    openpose_bin_path_and_configuration = 'CUDA_VISIBLE_DEVICES='+gpu+' '+openpose_bin+' --net_resolution "-1x512" --display 0 --scale_number 4 --scale_gap 0.25 --hand --hand_scale_number 6 --hand_scale_range 0.4 --render-pose=0 --model_folder /home/gts/projects/mvazquez/openpose/models'
    
    input_file = folder_in+"/"+vid+".mp4"
    output_folder = folder_out+'/'+vid
    
    command = openpose_bin_path_and_configuration+" --video "+input_file+" --write_json "+output_folder+"/"
    
    if os.path.exists(output_folder):
        print ("Remove folder: "+output_folder) 
        shutil.rmtree(output_folder)
        
    print ("Create folder: "+output_folder) 
    os.makedirs(output_folder)
    
    print(command)
    os.system(command)
    
def read_file_to_list(file):
    reader = pd.read_csv(file, header=None).values.tolist()
    return [i[0] for i in reader]

if __name__ == "__main__":
    
    openpose_bin = '/home/gts/projects/mvazquez/openpose/build/examples/openpose/openpose.bin'
    dataset_input = '../dataset/'
    
    part = str(args.set)
    split = str(args.split)
    gpu = str(args.gpu)
    
    folder_in = dataset_input+part+"/color"
    folder_out = part
    
    print (part, split, gpu)
    
    file = 'splits/'+part+'/op_'+part+'_split_'+split+".csv"
    print (file)
    
    lst_file = read_file_to_list(file)

    count=0
    for idx, file in enumerate(lst_file):
        print ("Processing: {}/{}".format((idx+1),len(lst_file)))
        print (file)
        exec_openpose(file, gpu=gpu)
