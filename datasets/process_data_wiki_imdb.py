import os
import tqdm
import argparse
import numpy as np
import utils

def get_args():
    parser = argparse.ArgumentParser(description="This script package imdb or wiki db to mxnet .rec format.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--rootpath", type=str, required=True,
                        help="root path to imdb or wiki crop db")
    parser.add_argument("--metafile", type=str, required=True, help='mat file' )
    parser.add_argument("--outfile", type=str, required=True,
                        help="output file name")
    parser.add_argument("--minscore", type=float,default=1.0,
                        help="face min score for filter noise")
    args = parser.parse_args()
    return args


def main():
    args=get_args()
    rootpath=args.rootpath
    outfile=args.outfile
    metafile =args.metafile
    min_score = args.minscore
    full_path, dob, gender, photo_taken, face_score, second_face_score, age=utils.get_meta(os.path.join(rootpath,'%s.mat'%metafile),metafile)

    total = 0

    label = []
    print("%d images " % len(face_score))
    for i in range(len(face_score)):
        if face_score[i] < min_score:
            continue

        if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
            continue

        if ~(0 <= age[i] <= 100):
            continue

        if np.isnan(gender[i]):
            continue
        fname=str(full_path[i][0])

        label.append([fname, age[i], gender[i]])
        total +=1

    with open(os.path.join(rootpath,outfile),'w') as f:
        for  line in label:
            f.write(line[0] + '\t' + str(line[1]) + '\t' + str(line[2]) + '\n')
    print("filter data")
    print("total: %d image" %(total))
    print('Done!!!')


if __name__ =='__main__':
    main()




    