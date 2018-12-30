import hashlib, csv, math, os, pickle, subprocess
from tqdm import tqdm

HEADER="Id,Label,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C20,C21,C22,C23,C24,C25,C26"

def open_with_first_line_skipped(path, skip=True):
    f = open(path)
    if not skip:
        return f
    next(f)
    return f

def hashstr(str, nr_bins):
    return int(hashlib.md5(str.encode('utf8')).hexdigest(), 16)%(nr_bins-1)+1

def gen_cate_feats(row, num_n, cate_n):
    feats = []
    for j in range(1, num_n+1):
        field = 'I{0}'.format(j)
        value = row[field]
        if value != '':
            value = int(value)
            if value > 2:
                value = int(math.log(float(value))**2)
            else:
                value = 'SP'+str(value)
        key = field + '$' + str(value)
        feats.append(key)
    for j in range(1, cate_n+1):
        field = 'C{0}'.format(j)
        value = row[field]
        key = field + '$' + value
        feats.append(key)
    return feats
    

def gen_num_feats(row, num_n, cate_n):
    feats = []
    for j in range(1, num_n+1):
        field = 'I{0}'.format(j)
        value = row[field]
        if value == '':
            value = 'mean'
        key = field + '$' + str(value)
        feats.append(key)
    for j in range(1, cate_n+1):
        field = 'C{0}'.format(j)
        value = row[field]
        key = field + '$' + value
        feats.append(key)
    return feats
    
def gen_feats(row, Type='category'):
    feats = []
    if Type != 'category':
        for j in range(1, 14):
            field = 'I{0}'.format(j)
            value = row[field]
            if value != '':
                value = float(value)
            else:
                value = 0
            # if value != '':
            #     value = int(value)
            #     if value > 2:
            #         value = int(math.log(float(value))**2)
            #     else:
            #         value = 'SP'+str(value)
            key = field + '$' + str(value)
            feats.append(key)
    if Type != 'numeric':
        for j in range(1, 27):
            field = 'C{0}'.format(j)
            value = row[field]
            key = field + '$' + value
            feats.append(key)
    return feats

def read_freqent_feats(threshold=10, rate=0.01,
                       data='criteo_offline0',
                       Type='num'):
    cate_emb_arr = {}
    num_mean_arr = {}
    frequent_feats = set()
    feat_freq = {}
    Samples = 0
    for row in tqdm(csv.DictReader(open('meta_data/' + data + '_cate_counts.csv'))):
        Samples += int(row['Total'])
        cate_emb_arr[row['Field']] = {}
        num_mean_arr[row['Field']] = {}
        feat_freq[row['Field']+'$'+row['Value']] = int(row['Total'])
    feat_freq = sorted(feat_freq.items(), key=lambda kv: -kv[1])
    total_feat = len(feat_freq)
    for k, v in feat_freq[:int((1-rate)*total_feat)]:
        if v >= threshold:
            frequent_feats.add(k)
    if Type == 'cate':
        return frequent_feats
    
    MaxIdx = {}
    for row in tqdm(csv.DictReader(open('meta_data/' + data + '_cate_counts.csv'))):
        feat = row['Field']+'$'+row['Value']
        if feat not in frequent_feats:
            feat = feat.split('$')[0]+'less'
            if feat not in cate_emb_arr[row['Field']]:
                cate_emb_arr[row['Field']][feat] = {'Cnt':eval(row['Total']), 'Idx':len(cate_emb_arr[row['Field']])} # 'Label':eval(row['Label']), 
            else:
                # cate_emb_arr[row['Field']][feat]['Label'] += eval(row['Label'])
                cate_emb_arr[row['Field']][feat]['Cnt'] += eval(row['Total'])
        else:
            cate_emb_arr[row['Field']][feat] = {'Cnt':eval(row['Total']), 'Idx':len(cate_emb_arr[row['Field']])} #'Label':eval(row['Label']),
    for row in tqdm(csv.DictReader(open('meta_data/' + data + '_num_counts.csv'))):
        num_mean_arr[row['Field']] = {'Sum':eval(row['Sum']), 'Cnt':eval(row['Cnt'])}
    for field in cate_emb_arr:
        MaxIdx[field] = len(cate_emb_arr[field])
        if field[0] == 'C' and field+'less' not in cate_emb_arr[field]:
            cate_emb_arr[field][field+'less'] = {'Cnt':0, 'Idx':MaxIdx[field]}
    return frequent_feats, cate_emb_arr, num_mean_arr, MaxIdx, Samples

def split(path, nr_thread, has_header):

    def open_with_header_witten(path, idx, header):
        f = open(path+'.__tmp__.{0}'.format(idx), 'w')
        if not has_header:
            return f 
        f.write(header)
        return f

    def calc_nr_lines_per_thread():
        nr_lines = int(list(subprocess.Popen('wc -l {0}'.format(path), shell=True, 
            stdout=subprocess.PIPE).stdout)[0].split()[0])
        if not has_header:
            nr_lines += 1 
        return math.ceil(float(nr_lines)/nr_thread)

    header = open(path).readline()

    nr_lines_per_thread = calc_nr_lines_per_thread()

    idx = 0
    f = open_with_header_witten(path, idx, header)
    for i, line in enumerate(open_with_first_line_skipped(path, has_header), start=1):
        if i%nr_lines_per_thread == 0:
            f.close()
            idx += 1
            f = open_with_header_witten(path, idx, header)
        f.write(line)
    f.close()

def parallel_convert(cvt_path, arg_paths, nr_thread):

    workers = []
    for i in range(nr_thread):
        cmd = '{0}'.format(os.path.join(cvt_path))
        for path in arg_paths:
            cmd += ' {0}'.format(path+'.__tmp__.{0}'.format(i))
        worker = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        workers.append(worker)
    for worker in workers:
        worker.communicate()

def cat(path, nr_thread):
    
    if os.path.exists(path):
        os.remove(path)
    for i in range(nr_thread):
        cmd = 'cat {svm}.__tmp__.{idx} >> {svm}'.format(svm=path, idx=i)
        p = subprocess.Popen(cmd, shell=True)
        p.communicate()

def delete(path, nr_thread):
    
    for i in range(nr_thread):
        os.remove('{0}.__tmp__.{1}'.format(path, i))

