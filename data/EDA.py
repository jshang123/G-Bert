# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import dill
from random import shuffle
import random
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'data'))
    print(os.getcwd())
except:
    pass

# %%
import pandas as pd
from collections import defaultdict
import numpy as np

med_file = 'PRESCRIPTIONS.csv'
diag_file = 'DIAGNOSES_ICD.csv'
ndc2atc_file = 'ndc2atc_level4.csv'
ddi_file = 'drug-DDI.csv'
cid_atc = 'drug-atc.csv'
patient_info_file = './gather_firstday.csv'


def process_med():
    print('process_med')
    med_pd = pd.read_csv(med_file, dtype={'NDC': 'category'})
    # filter
    med_pd.drop(columns=['ROW_ID', 'DRUG_TYPE', 'DRUG_NAME_POE', 'DRUG_NAME_GENERIC',
                         'FORMULARY_DRUG_CD', 'GSN', 'PROD_STRENGTH', 'DOSE_VAL_RX',
                         'DOSE_UNIT_RX', 'FORM_VAL_DISP', 'FORM_UNIT_DISP', 'FORM_UNIT_DISP',
                         'ROUTE', 'ENDDATE', 'DRUG'], axis=1, inplace=True)
    med_pd.drop(index=med_pd[med_pd['NDC'] == '0'].index, axis=0, inplace=True)
    med_pd.fillna(method='pad', inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    med_pd['ICUSTAY_ID'] = med_pd['ICUSTAY_ID'].astype('int64')
    med_pd['STARTDATE'] = pd.to_datetime(
        med_pd['STARTDATE'], format='%Y-%m-%d %H:%M:%S')
    med_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID',
                           'ICUSTAY_ID', 'STARTDATE'], inplace=True)
    med_pd = med_pd.reset_index(drop=True)

    def filter_first24hour_med(med_pd):
        med_pd_new = med_pd.drop(columns=['NDC'])
        med_pd_new = med_pd_new.groupby(
            by=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID']).head([1]).reset_index(drop=True)
        med_pd_new = pd.merge(med_pd_new, med_pd, on=[
                              'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE'])
        med_pd_new = med_pd_new.drop(columns=['STARTDATE'])
        return med_pd_new
    med_pd = filter_first24hour_med(med_pd)  # or next line
#     med_pd = med_pd.drop(columns=['STARTDATE'])

    med_pd = med_pd.drop(columns=['ICUSTAY_ID'])
    med_pd = med_pd.drop_duplicates()

    return med_pd.reset_index(drop=True)


def process_diag():
    print('process_diag')

    diag_pd = pd.read_csv(diag_file)
    diag_pd.dropna(inplace=True)
    diag_pd.drop(columns=['SEQ_NUM', 'ROW_ID'], inplace=True)
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID'], inplace=True)
    return diag_pd.reset_index(drop=True)


def process_side():
    print('process_side')

    side_pd = pd.read_csv(patient_info_file)
    # just use demographic information to avoid future information leak such as lab test and lab measurements
    side_pd = side_pd[['subject_id', 'hadm_id', 'icustay_id',
                       'gender_male', 'admission_type', 'first_icu_stay', 'admission_age',
                       'ethnicity', 'weight', 'height']]

    # process side_information
    side_pd = side_pd.dropna(thresh=4)
    side_pd.fillna(side_pd.mean(), inplace=True)
    side_pd = side_pd.groupby(by=['subject_id', 'hadm_id']).head(
        [1]).reset_index(drop=True)
    side_pd = pd.concat(
        [side_pd, pd.get_dummies(side_pd['ethnicity'])], axis=1)
    side_pd.drop(columns=['ethnicity', 'icustay_id'], inplace=True)
    side_pd.rename(columns={'subject_id': 'SUBJECT_ID',
                            'hadm_id': 'HADM_ID'}, inplace=True)
    return side_pd.reset_index(drop=True)


def ndc2atc4(med_pd):
    with open('ndc2rxnorm_mapping.txt', 'r') as f:
        ndc2rxnorm = eval(f.read())
    med_pd['RXCUI'] = med_pd['NDC'].map(ndc2rxnorm)
    med_pd.dropna(inplace=True)

    rxnorm2atc = pd.read_csv('ndc2atc_level4.csv')
    rxnorm2atc = rxnorm2atc.drop(columns=['YEAR', 'MONTH', 'NDC'])
    rxnorm2atc.drop_duplicates(subset=['RXCUI'], inplace=True)
    med_pd.drop(index=med_pd[med_pd['RXCUI'].isin(
        [''])].index, axis=0, inplace=True)

    med_pd['RXCUI'] = med_pd['RXCUI'].astype('int64')
    med_pd = med_pd.reset_index(drop=True)
    med_pd = med_pd.merge(rxnorm2atc, on=['RXCUI'])
    med_pd.drop(columns=['NDC', 'RXCUI'], inplace=True)
#     med_pd = med_pd.rename(columns={'ATC4':'NDC'})
    med_pd['ATC4'] = med_pd['ATC4'].map(lambda x: x[:5])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)
    return med_pd


def filter_pro(pro_pd):
    pro_count = pro_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(
        columns={0: 'count'}).sort_values(by=['count'], ascending=False).reset_index(drop=True)
    pro_pd = pro_pd[pro_pd['ICD9_CODE'].isin(
        pro_count.loc[:1000, 'ICD9_CODE'])]

    return pro_pd.reset_index(drop=True)


def filter_diag(diag_pd, num=128):
    print('filter diag')
    diag_count = diag_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(
        columns={0: 'count'}).sort_values(by=['count'], ascending=False).reset_index(drop=True)
    diag_pd = diag_pd[diag_pd['ICD9_CODE'].isin(
        diag_count.loc[:num, 'ICD9_CODE'])]

    return diag_pd.reset_index(drop=True)


def filter_med(med_pd):
    med_count = med_pd.groupby(by=['ATC4']).size().reset_index().rename(columns={
        0: 'count'}).sort_values(by=['count'], ascending=False).reset_index(drop=True)
    med_pd = med_pd[med_pd['ATC4'].isin(med_count.loc[:299, 'ATC4'])]

    return med_pd.reset_index(drop=True)

# visit filter


def filter_by_visit_range(data_pd, v_range=(1, 2)):
    a = data_pd[['SUBJECT_ID', 'HADM_ID']].groupby(
        by='SUBJECT_ID')['HADM_ID'].unique().reset_index()
    a['HADM_ID_Len'] = a['HADM_ID'].map(lambda x: len(x))
    a = a[(a['HADM_ID_Len'] >= v_range[0]) & (a['HADM_ID_Len'] < v_range[1])]
    data_pd_filter = a.reset_index(drop=True)
    data_pd = data_pd.merge(
        data_pd_filter[['SUBJECT_ID']], on='SUBJECT_ID', how='inner')
    return data_pd.reset_index(drop=True)


def process_all(visit_range=(1, 2)):
    # get med and diag (visit>=2)
    med_pd = process_med()
    med_pd = ndc2atc4(med_pd)
#     med_pd = filter_300_most_med(med_pd)
    med_pd = filter_by_visit_range(med_pd, visit_range)

    diag_pd = process_diag()
    diag_pd = filter_diag(diag_pd, num=1999)

#     side_pd = process_side()

#     pro_pd = process_procedure()
#     pro_pd = filter_1000_most_pro(pro_pd)

    med_pd_key = med_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    diag_pd_key = diag_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
#     pro_pd_key = pro_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
#     side_pd_key = side_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()

    combined_key = med_pd_key.merge(
        diag_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
#     combined_key = combined_key.merge(pro_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
#     combined_key = combined_key.merge(side_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    diag_pd = diag_pd.merge(
        combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    med_pd = med_pd.merge(
        combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
#     side_pd = side_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
#     pro_pd = pro_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    # flatten and merge
    diag_pd = diag_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])[
        'ICD9_CODE'].unique().reset_index()
    med_pd = med_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])[
        'ATC4'].unique().reset_index()
#     pro_pd = pro_pd.groupby(by=['SUBJECT_ID','HADM_ID'])['ICD9_CODE'].unique().reset_index().rename(columns={'ICD9_CODE':'PRO_CODE'})
    diag_pd['ICD9_CODE'] = diag_pd['ICD9_CODE'].map(lambda x: list(x))
    med_pd['ATC4'] = med_pd['ATC4'].map(lambda x: list(x))
#     pro_pd['PRO_CODE'] = pro_pd['PRO_CODE'].map(lambda x: list(x))
    data = diag_pd.merge(med_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
#     data = data.merge(side_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
#     data = data.merge(pro_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
#     data['ICD9_CODE_Len'] = data['ICD9_CODE'].map(lambda x: len(x))
#     data['NDC_Len'] = data['NDC'].map(lambda x: len(x))
    return data


def filter_patient(data, dx_range=(2, np.inf), rx_range=(2, np.inf)):
    print('filter_patient')

    drop_subject_ls = []
    for subject_id in data['SUBJECT_ID'].unique():
        item_data = data[data['SUBJECT_ID'] == subject_id]

        for index, row in item_data.iterrows():
            dx_len = len(list(row['ICD9_CODE']))
            rx_len = len(list(row['ATC4']))
            if dx_len < dx_range[0] or dx_len > dx_range[1]:
                drop_subject_ls.append(subject_id)
                break
            if rx_len < rx_range[0] or rx_len > rx_range[1]:
                drop_subject_ls.append(subject_id)
                break
    data.drop(index=data[data['SUBJECT_ID'].isin(
        drop_subject_ls)].index, axis=0, inplace=True)
    return data.reset_index(drop=True)


def statistics(data):
    print('#patients ', data['SUBJECT_ID'].unique().shape)
    print('#clinical events ', len(data))

    diag = data['ICD9_CODE'].values
    med = data['ATC4'].values

    unique_diag = set([j for i in diag for j in list(i)])
    unique_med = set([j for i in med for j in list(i)])

    print('#diagnosis ', len(unique_diag))
    print('#med ', len(unique_med))

    avg_diag = 0
    avg_med = 0
    max_diag = 0
    max_med = 0
    cnt = 0
    max_visit = 0
    avg_visit = 0

    for subject_id in data['SUBJECT_ID'].unique():
        item_data = data[data['SUBJECT_ID'] == subject_id]
        x = []
        y = []
        visit_cnt = 0
        for index, row in item_data.iterrows():
            visit_cnt += 1
            cnt += 1
            x.extend(list(row['ICD9_CODE']))
            y.extend(list(row['ATC4']))
        x = set(x)
        y = set(y)
        avg_diag += len(x)
        avg_med += len(y)
        avg_visit += visit_cnt
        if len(x) > max_diag:
            max_diag = len(x)
        if len(y) > max_med:
            max_med = len(y)
        if visit_cnt > max_visit:
            max_visit = visit_cnt

    print('#avg of diagnoses ', avg_diag / cnt)
    print('#avg of medicines ', avg_med / cnt)
    print('#avg of vists ', avg_visit / len(data['SUBJECT_ID'].unique()))

    print('#max of diagnoses ', max_diag)
    print('#max of medicines ', max_med)
    print('#max of visit ', max_visit)


def run(visit_range=(1, 2)):
    data = process_all(visit_range)
    data = filter_patient(data)

    # unique code save
    diag = data['ICD9_CODE'].values
    med = data['ATC4'].values
    unique_diag = set([j for i in diag for j in list(i)])
    unique_med = set([j for i in med for j in list(i)])

    return data, unique_diag, unique_med


def load_gamenet_multi_visit_data(file_name='data_gamenet.pkl'):
    data = pd.read_pickle(file_name)
    data.rename(columns={'NDC': 'ATC4'}, inplace=True)
    data.drop(columns=['PRO_CODE', 'NDC_Len'], axis=1, inplace=True)

    # unique code save
    diag = data['ICD9_CODE'].values
    med = data['ATC4'].values
    unique_diag = set([j for i in diag for j in list(i)])
    unique_med = set([j for i in med for j in list(i)])
    return data, unique_diag, unique_med


def load_gamenet_multi_visit_data_with_pro(file_name='data_gamenet.pkl'):
    data = pd.read_pickle(file_name)
    data.rename(columns={'NDC': 'ATC4'}, inplace=True)
    data.drop(columns=['NDC_Len'], axis=1, inplace=True)

    # unique code save
    diag = data['ICD9_CODE'].values
    med = data['ATC4'].values
    pro = data['PRO_CODE'].values
    unique_diag = set([j for i in diag for j in list(i)])
    unique_med = set([j for i in med for j in list(i)])
    unique_pro = set([j for i in pro for j in list(i)])

    return data, unique_pro, unique_diag, unique_med


def main():
    print('-'*20 + '\ndata-single processing')
    data_single_visit, diag1, med1 = run(visit_range=(1, 2))
    print('-'*20 + '\ndata-multi processing ')
    data_multi_visit, pro, diag2, med2 = load_gamenet_multi_visit_data_with_pro()
#     med_diag_pair = gen_med_diag_pair(data)

    unique_diag = diag1 | diag2
    unique_med = med1 | med2
    with open('dx-vocab.txt', 'w') as fout:
        for code in unique_diag:
            fout.write(code + '\n')
    with open('rx-vocab.txt', 'w') as fout:
        for code in unique_med:
            fout.write(code + '\n')

    with open('rx-vocab-multi.txt', 'w') as fout:
        for code in med2:
            fout.write(code + '\n')
    with open('dx-vocab-multi.txt', 'w') as fout:
        for code in diag2:
            fout.write(code + '\n')
    with open('px-vocab-multi.txt', 'w') as fout:
        for code in pro:
            fout.write(code + '\n')

    # save data
    data_single_visit.to_pickle('data-single-visit.pkl')
    data_multi_visit.to_pickle('data-multi-visit.pkl')

#     med_diag_pair.to_pickle('med_diag.pkl')
#     print('med2diag len:', len(med_diag_pair))

    print('-'*20 + '\ndata-single stat')
    statistics(data_single_visit)
    print('-'*20 + '\ndata_multi stat')
    statistics(data_multi_visit)

    return data_single_visit, data_multi_visit


data_single_visit, data_multi_visit = main()
data_multi_visit.head(10)


# %%
# split train, eval and test dataset
random.seed(1203)


def split_dataset(data_path='data-multi-visit.pkl'):
    data = pd.read_pickle(data_path)
    sample_id = data['SUBJECT_ID'].unique()

    random_number = [i for i in range(len(sample_id))]
#     shuffle(random_number)

    train_id = sample_id[random_number[:int(len(sample_id)*2/3)]]
    eval_id = sample_id[random_number[int(
        len(sample_id)*2/3): int(len(sample_id)*5/6)]]
    test_id = sample_id[random_number[int(len(sample_id)*5/6):]]

    def ls2file(list_data, file_name):
        with open(file_name, 'w') as fout:
            for item in list_data:
                fout.write(str(item) + '\n')

    ls2file(train_id, 'train-id.txt')
    ls2file(eval_id, 'eval-id.txt')
    ls2file(test_id, 'test-id.txt')

    print('train size: %d, eval size: %d, test size: %d' %
          (len(train_id), len(eval_id), len(test_id)))


split_dataset()


# %%
# generate ehr graph for gamenet


def generate_ehr_graph():
    data_multi = pd.read_pickle('data-multi-visit.pkl')
    data_single = pd.read_pickle('data-single-visit.pkl')

    rx_voc_size = 0
    rx_voc = {}
    with open('rx-vocab.txt', 'r') as fin:
        for line in fin:
            rx_voc[line.rstrip('\n')] = rx_voc_size
            rx_voc_size += 1

    ehr_adj = np.zeros((rx_voc_size, rx_voc_size))

    for idx, row in data_multi.iterrows():
        med_set = list(map(lambda x: rx_voc[x], row['ATC4']))
        for i, med_i in enumerate(med_set):
            for j, med_j in enumerate(med_set):
                if j <= i:
                    continue
                ehr_adj[med_i, med_j] = 1
                ehr_adj[med_j, med_i] = 1

    for idx, row in data_single.iterrows():
        med_set = list(map(lambda x: rx_voc[x], row['ATC4']))
        for i, med_i in enumerate(med_set):
            for j, med_j in enumerate(med_set):
                if j <= i:
                    continue
                ehr_adj[med_i, med_j] = 1
                ehr_adj[med_j, med_i] = 1

    print('avg med for one ', np.mean(np.sum(ehr_adj, axis=-1)))

    return ehr_adj


ehr_adj = generate_ehr_graph()
dill.dump(ehr_adj, open('ehr_adj.pkl', 'wb'))


# %%
# max len medical codes
data = data_multi_visit

max_len = 0
for subject_id in data['SUBJECT_ID'].unique():
    item_df = data[data['SUBJECT_ID'] == subject_id]
    len_tmp = 0
    for index, row in item_df.iterrows():
        len_tmp += (len(row['ICD9_CODE']) + len(row['ATC4']))
    if len_tmp > max_len:
        max_len = len_tmp
print(max_len)


# %%
print(max_len)


# %%
pd.read_pickle(file_name)


# %%
data.rename(columns={'NDC': 'ATC4'}, inplace=True)
data.drop(columns=['PRO_CODE', 'NDC_Len'], axis=1, inplace=True)


# %%
data.shape


# %%
data_dir = './data/'

print('multi visit')
multi_file = data_dir + 'data-multi-visit.pkl'
multi_pkl = pd.read_pickle(multi_file)
multi_pkl.iloc[0, 4:]

# %%
# stat
rx_cnt_ls = []
dx_cnt_ls = []
visit_cnt_ls = []
for subject_id in multi_pkl['SUBJECT_ID'].unique():
    visit_cnt = 0
    for idx, visit in multi_pkl[multi_pkl['SUBJECT_ID'] == subject_id].iterrows():
        rx_cnt_ls.append(len(visit['ATC4']))
        dx_cnt_ls.append(len(visit['ICD9_CODE']))
        visit_cnt += 1
    visit_cnt_ls.append(visit_cnt)

print('mean')
print('dx', np.mean(dx_cnt_ls))
print('rx', np.mean(rx_cnt_ls))
print('visit', np.mean(visit_cnt_ls))

print('max')
print('dx', np.max(dx_cnt_ls))
print('rx', np.max(rx_cnt_ls))
print('visit', np.max(visit_cnt_ls))

print('min')
print('dx', np.min(dx_cnt_ls))
print('rx', np.min(rx_cnt_ls))
print('visit', np.min(visit_cnt_ls))


print('single visit')
# %%
single_file = data_dir + 'data-single-visit.pkl'
single_pkl = pd.read_pickle(single_file)
single_pkl.head()
# %%

rx_cnt_ls = []
dx_cnt_ls = []
visit_cnt_ls = []
for subject_id in single_pkl['SUBJECT_ID'].unique():
    visit_cnt = 0
    for idx, visit in single_pkl[single_pkl['SUBJECT_ID'] == subject_id].iterrows():
        rx_cnt_ls.append(len(visit['ATC4']))
        dx_cnt_ls.append(len(visit['ICD9_CODE']))
        visit_cnt += 1
    visit_cnt_ls.append(visit_cnt)

print('mean')
print('dx', np.mean(dx_cnt_ls))
print('rx', np.mean(rx_cnt_ls))
print('visit', np.mean(visit_cnt_ls))

print('max')
print('dx', np.max(dx_cnt_ls))
print('rx', np.max(rx_cnt_ls))
print('visit', np.max(visit_cnt_ls))

print('min')
print('dx', np.min(dx_cnt_ls))
print('rx', np.min(rx_cnt_ls))
print('visit', np.min(visit_cnt_ls))

# multi visit
# mean
# dx 13.640849760255728
# rx 13.930074587107086
# visit 2.3647244094488187
# max
# dx 39
# rx 36
# visit 29
# min
# dx 1
# rx 1
# visit 1
# single visit
# mean
# dx 10.820458611156285
# rx 13.759277931370955
# visit 1.0
# max
# dx 39
# rx 52
# visit 1
# min
# dx 2
# rx 2
# visit 1


# %%
data_dir = './data/'

single_pkl.head()


# %%
