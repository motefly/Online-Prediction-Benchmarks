python utils/count.py /share_data/Data/criteo_labeled.csv > fc.trva.t10.txt
python utils/parallelizer-b.py -s 24 'python utils/pre-b.py' data/criteo_labeled.csv data/criteo_labeled.ffm
