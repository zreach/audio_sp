import os
import random
import numpy as np
import argparse
import logging
import soundfile as sf
import json 


def create_one_dir(input_dir, output_dir, nums_file, state):
    if(not os.path.exists(output_dir)):
        os.mkdir(output_dir)
    min_length = 3 #最低秒数，筛选过短音频


    a_wavList = []
    b_wavList = []
    dir_list = os.listdir(input_dir)
    random.shuffle(dir_list)
    a_spk_lst,b_spk_list = dir_list[:len(dir_list)//2],dir_list[len(dir_list)//2:]
    for root, _, files in os.walk(input_dir):

        # print(root)
        for file in files: 
            if (file.endswith('WAV') or file.endswith('wav') or file.endswith('flac')):

                wavFile = os.path.join(root, file)
                data, sr = sf.read(wavFile)
                if len(data.shape) != 1:
                    data = np.mean(data,axis=1).reshape(-1)
                    sf.write(wavFile,data,sr)

                if data.shape[0] < sr * min_length:
                    pass
                else:
                    for spk in a_spk_lst:

                        if (os.path.join(input_dir,spk) in root):
                            a_wavList.append(wavFile)
                    for spk in b_spk_list:
                        if (os.path.join(input_dir,spk) in root):
                            b_wavList.append(wavFile)

    random.shuffle(a_wavList)
    random.shuffle(b_wavList)
    print(len(a_wavList))
    print(len(b_wavList))
    mix_list = []
    
    for i in range(len(a_wavList)):
        for j in range(len(b_wavList)):
            mix_list.append([a_wavList[i],b_wavList[j]])
    random.shuffle(mix_list)
    mix_list = mix_list[:nums_file]
    a_wavList,b_wavList = [mix[0] for mix in mix_list],[mix[1] for mix in mix_list]
    print(len(a_wavList))
    
    s1_path = os.path.join(output_dir,'s1.json' )
    with open(s1_path,'w') as fp:
        json.dump(a_wavList,fp,indent=4)
    s2_path = os.path.join(output_dir,'s2.json' )
    with open(s2_path,'w') as fp:
        json.dump(b_wavList,fp,indent=4)

def CreateFiles(input_dir, output_dir, nums_file):
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    a_wavList = []
    b_wavList = []
        
    # print(os.listdir(input_dir))

    
    name_dic = {'TRAIN':'tr','DEV':'cv','TEST':'tt'}
    percent_dic = {'TRAIN':1 ,'DEV':0.1,'TEST':0.1} #训练、验证、测试集比例
    for state in ['TRAIN','DEV','TEST']:
        in_dir = os.path.join(input_dir,state)
        out_dir = os.path.join(output_dir,name_dic[state])
        create_one_dir(input_dir,out_dir,int(nums_file*percent_dic[state]),state)


def run(args):
    logging.basicConfig(level=logging.INFO)
        
    input_dir =args.input_dir
    output_dir = args.output_dir
    state = args.state
    nums_file = args.nums_files
    CreateFiles(input_dir, output_dir, nums_file, state)
    logging.info("Done create initial data pair")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Command to make separation dataset'
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to input data directory"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help='Path ot output data directory'
    )
    parser.add_argument(
        "--nums_files",
        type=int,
        help='Path ot output data directory'
    )
    parser.add_argument(
        "--state",
        type=str,
        help='Whether create train or test data directory'
    )
    args = parser.parse_args()
    run(args)
