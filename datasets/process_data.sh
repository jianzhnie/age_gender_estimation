#!/bin/sh

python process_data_wiki_imdb.py --rootpath '/media/dm/d/data/age_data/imdb_crop' \
                                --metafile 'imdb' \
                                --outfile  'imdbfilelist.txt'

python process_data_wiki_imdb.py --rootpath '/media/dm/d/data/age_data/wiki_crop' \
                                --metafile 'wiki' \
                                --outfile  'wikifilelist.txt'