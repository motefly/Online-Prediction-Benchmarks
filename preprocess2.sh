# python utils/split_train.py data/criteo/criteo_labeled.csv  data/criteo/criteo_offline.csv
# python utils/count.py /share_data/Data/criteo_offline0.csv
# python utils/parallelizer-b.py -s 24 'python utils/pre-b.py' data/criteo_labeled.csv data/criteo_labeled.ffm
python utils/count.py data/criteo/criteo_2random0.csv criteo_2random --thres 10
python utils/pre-num.py data/criteo/criteo_2random0.csv data/criteo/criteo_2random0.num -d criteo_2random
python utils/pre-num.py data/criteo/criteo_2random1.csv data/criteo/criteo_2random1.num -d criteo_2random -p test
python utils/pre-cate.py data/criteo/criteo_2random0.csv data/criteo/criteo_2random0.cate -d criteo_2random
python utils/pre-cate.py data/criteo/criteo_2random1.csv data/criteo/criteo_2random1.cate -d criteo_2random -p test

# python utils/split_train.py data/criteo/sample.csv data/criteo/sample_offline.csv
# python utils/count.py data/criteo/sample_offline0.csv sample_offline --thres 1
# python utils/pre-num.py data/criteo/sample_offline0.csv data/criteo/sample_offline0.num -d sample_offline -t 1
# python utils/pre-num.py data/criteo/sample_offline1.csv data/criteo/sample_offline1.num -d sample_offline -p test -t 1
# python utils/pre-idv.py data/criteo/sample_offline0.csv data/criteo/sample_offline0.idv -d sample_offline -t 1
# python utils/pre-idv.py data/criteo/sample_offline1.csv data/criteo/sample_offline1.idv -d sample_offline -p test -t 1
