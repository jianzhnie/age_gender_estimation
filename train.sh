#!/bin/sh

python main.py --data_dir /media/dm/d/data/age_data/imdb_crop \
               --filename_list /media/dm/d/data/age_data/imdb_crop/imdbfilelist.txt \
               --test_data_dir /media/dm/d/data/age_data/wiki_crop \
               --test_filename_list /media/dm/d/data/age_data/wiki_crop/wikifilelist.txt \
               --batch_size 64 --lr 0.01 \
               --gpu 0  --print-freq 10