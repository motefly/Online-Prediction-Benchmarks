# python utils/split_train.py data/criteo/criteo_labeled.csv  data/criteo/criteo_offline.csv
# python utils/count.py /share_data/Data/criteo_offline0.csv
# python utils/parallelizer-b.py -s 24 'python utils/pre-b.py' data/criteo_labeled.csv data/criteo_labeled.ffm
python utils/count.py data/criteo/criteo_offline0.csv criteo_offline --thres 10
python utils/pre-num.py data/criteo/criteo_offline0.csv data/criteo/criteo_offline0.num -d criteo_offline
python utils/pre-num.py data/criteo/criteo_offline1.csv data/criteo/criteo_offline1.num -d criteo_offline -p test
python utils/pre-cate.py data/criteo/criteo_offline0.csv data/criteo/criteo_offline0.cate -d criteo_offline
python utils/pre-cate.py data/criteo/criteo_offline1.csv data/criteo/criteo_offline1.cate -d criteo_offline -p test
