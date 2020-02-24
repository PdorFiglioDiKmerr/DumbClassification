import pandas as pd
import numpy as np
import json
import datetime
from Label import labelling
from Label import labelling2
from InterStatistics import inter_statistic
import os
import traceback 

def kbps(series):
    return series.sum()*8/1024

def zeroes_count(series):
    a = series[series == 0].count()
    if np.isnan(a):
        return 0
    else:
        return a

def value_label(series):

    value = series.value_counts()
    try:
        return value.index[0]
    except:
        pass

def p25(x):
    return (x.quantile(0.25)* 0.01)

def p50(x):
    return (x.quantile(0.50)* 0.01)

def p75(x):
    return (x.quantile(0.75)* 0.01)

def max_min_diff(series):
    return series.max() - series.min()



def json2stat (dict_flow_data, pcap_path, name, screen = None, quality = None):
    try:
        #print ("Sono Dentro")
        LEN_DROP = 0
        dict_flow_data, LEN_DROP = inter_statistic (dict_flow_data, LEN_DROP)
        #print ("Statistiche: {}".format(dict_flow_data.keys()))
        dict_flow_data = labelling (dict_flow_data, screen, quality)
        dict_flow_data = labelling2(dict_flow_data, screen, quality)
        #print ("Statistiche2: {}".format(dict_flow_data.keys()))        
        df_train = pd.DataFrame()
        #sprint("PID: {},  info: {}".format(os.getpid(),dict_flow_data.keys()))
        for flow_id in dict_flow_data.keys():
            dict_flow_data[flow_id]["timestamps"] = pd.to_datetime(dict_flow_data[flow_id]["timestamps"], unit = 's')
            dict_flow_data[flow_id].set_index('timestamps', inplace = True)
            dict_flow_data[flow_id] = dict_flow_data[flow_id].dropna()
            train = dict_flow_data[flow_id].resample('s').agg({'interarrival' : ['std', 'mean', p25, p50, p75, max_min_diff], 'len_udp' : ['std', 'mean', 'count', kbps, p25, p50, p75, max_min_diff], \
                'interlength_udp' : ['mean', p25, p50, p75, max_min_diff], 'rtp_interarrival' : ['std', 'mean', zeroes_count, max_min_diff] ,\
                "inter_time_sequence": ['std', 'mean', p25, p50, p75, max_min_diff] ,"label": [value_label],  "label2": [value_label]})

            df_train = pd.concat([df_train, train])
        dataset_dropped = df_train.dropna()
        dataset_dropped.reset_index(inplace = True, drop = True)
        #print("col: {}\nindex {}".format(dataset_dropped.columns,dataset_dropped.index))
        new_header = []
        for h in dataset_dropped.columns:
            new_header.append(h[0] + "_" + h[1])
        #   dataset_dropped.columns = dataset_dropped.columns.droplevel()
        dataset_dropped.columns = new_header
        dataset_dropped = dataset_dropped.rename(columns={'label_value_label': 'label'})
        dataset_dropped = dataset_dropped.rename(columns={'label2_value_label': 'label2'})
        dataset_dropped = dataset_dropped.rename(columns={'len_udp_kbps': 'kbps'})
        dataset_dropped = dataset_dropped.rename(columns={'len_udp_count': 'num_packets'})
        dataset_dropped = dataset_dropped.rename(columns={'rtp_interarrival_std': 'rtp_inter_timestamp_std'})
        dataset_dropped = dataset_dropped.rename(columns={'rtp_interarrival_mean': 'rtp_inter_timestamp_mean'})
        dataset_dropped = dataset_dropped.rename(columns={'rtp_interarrival_zeroes_count': 'rtp_inter_timestamp_num_zeros'})
        pcap_path = os.path.join(pcap_path, name)
        with open(pcap_path + ".csv", "w") as file:
            dataset_dropped.to_csv( file, index = False)
        return
    except Exception as e:
        print("Sto fallendo qui " + str(e))
