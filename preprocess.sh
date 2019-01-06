# python utils/split_train.py data/criteo/criteo_labeled.csv  data/criteo/criteo_offline.csv
# python utils/count.py /share_data/Data/criteo_offline0.csv
# python utils/parallelizer-b.py -s 24 'python utils/pre-b.py' data/criteo_labeled.csv data/criteo_labeled.ffm
python utils/count.py data/criteo/tinycriteo_expr0.csv tinycriteo_expr --thres 10
python utils/pre-num.py data/criteo/tinycriteo_expr0.csv data/criteo/tinycriteo_expr0.num -d tinycriteo_expr
python utils/pre-num.py data/criteo/tinycriteo_expr1.csv data/criteo/tinycriteo_expr1.num -d tinycriteo_expr -p test
python utils/pre-cate.py data/criteo/tinycriteo_expr0.csv data/criteo/tinycriteo_expr0.cate -d tinycriteo_expr
python utils/pre-cate.py data/criteo/tinycriteo_expr1.csv data/criteo/tinycriteo_expr1.cate -d tinycriteo_expr -p test

# python utils/split_train.py data/criteo/sample.csv data/criteo/sample_offline.csv
# python utils/count.py data/criteo/sample_offline0.csv sample_offline --thres 1
# python utils/pre-num.py data/criteo/sample_offline0.csv data/criteo/sample_offline0.num -d sample_offline -t 1
# python utils/pre-num.py data/criteo/sample_offline1.csv data/criteo/sample_offline1.num -d sample_offline -p test -t 1
# python utils/pre-idv.py data/criteo/sample_offline0.csv data/criteo/sample_offline0.idv -d sample_offline -t 1
# python utils/pre-idv.py data/criteo/sample_offline1.csv data/criteo/sample_offline1.idv -d sample_offline -p test -t 1

# python utils/count.py data/criteo/test_expr0.csv test_expr --thres 1 --numC 2 --numI 2
# python utils/pre-num.py data/criteo/test_expr0.csv data/criteo/test_expr0.num -d test_expr -t 1 --numC 2 --numI 2
# python utils/pre-num.py data/criteo/test_expr1.csv data/criteo/test_expr1.num -d test_expr -p test -t 1 --numC 2 --numI 2
# python utils/pre-cate.py data/criteo/test_expr0.csv data/criteo/test_expr0.cate -d test_expr -t 1 --numC 2 --numI 2
# python utils/pre-cate.py data/criteo/test_expr1.csv data/criteo/test_expr1.cate -d test_expr -p test -t 1 --numC 2 --numI 2
