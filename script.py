#!/usr/bin/env python
# coding=utf-8
import subprocess
import os
import argparse
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for compute dice.')
    parser.add_argument('-dir1', type=str, default=r'C:\Users\admin\Desktop\test_data_gd')
    parser.add_argument('-dir2', type=str, default=r"C:\Users\admin\Desktop\submit_image")
    args = parser.parse_args()

    files_list1 = next(os.walk(args.dir1))[2]
    files_list2 = next(os.walk(args.dir2))[2]
    print('scar','   scar+edema')
    scardice = []
    scaredemadice = []
    for idx, f in enumerate(files_list1):
        cmd = 'zxhCardMyoPSEvaluate.exe -evaps {} {} 1'.format(args.dir1 + '/' + files_list1[idx], args.dir2 + '/' + files_list2[idx])
        
        # Way 1
        # n = os.system(cmd)

        # Way 2
        result = os.popen(cmd)
        text = result.read()
        scaredema_dice = float(text.split()[-1])
        if text.split()[-2]== '-1.#IND00':
            scar_dice = 0
        else:
            scar_dice = float(text.split()[-2])
        scaredemadice.append(scaredema_dice)
        scardice.append(scar_dice)
        print(scar_dice, scaredema_dice)

    scar_avg = np.mean(scardice)
    scaredema_avg = np.mean(scaredemadice)
    print('\nscar_avg:', scar_avg,'\nscar+edema_avg:', scaredema_avg)