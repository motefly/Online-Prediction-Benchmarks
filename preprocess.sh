python utils/split_train.py data/criteo/criteo_labeled.csv  data/criteo/criteo_offline.csv
python utils/count.py /share_data/Data/criteo_offline0.csv
python utils/parallelizer-b.py -s 24 'python utils/pre-b.py' data/criteo_labeled.csv data/criteo_labeled.ffm
