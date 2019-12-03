#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import json
import matplotlib.pyplot as plt
from json import JSONDecoder, JSONDecodeError
import sys
import numpy as np
import re

from plotting import plot_stuff

#%%
LEN_DROP = 0
counter_ssrc = 0

def decode_stacked(document, pos=0, decoder=JSONDecoder()):
    NOT_WHITESPACE = re.compile(r'[^\s]')
    while True:
        match = NOT_WHITESPACE.search(document, pos)
        if not match:
            return
        pos = match.start()

        try:
            obj, pos = decoder.raw_decode(document, pos)
        except JSONDecodeError:
            # do something sensible if there's some error
            raise
        yield obj
        
        
        
#Open Tshark and run command to turn pcap to json
def pcap_to_json(source_pcap):
    import subprocess
        
    # Retrive all STUN packets
    command = ['tshark', '-r', source_pcap, '-l', '-n', '-T', 'ek', '-Y (stun)']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=None, encoding = 'utf-8', )
    try:
        output, error = process.communicate()
    except Exception as e:
        print ("Errore in pcap_to_json " + str(e))
        process.kill()
    
    # I've got all STUN packets: need to find which ports are used by RTP
    used_port = set()
    for obj in decode_stacked(output):
        try:
            if 'index' in obj.keys():
                continue
            if 'stun' in obj['layers'].keys() and "0x00000101" in obj['layers']["stun"]["stun_stun_type"]:          #0x0101 means success
                used_port.add(obj['layers']["udp"]["udp_udp_srcport"])
                used_port.add(obj['layers']["udp"]["udp_udp_dstport"])
        except:
            pass
    command = ['tshark', '-r', source_pcap, '-l', '-n', '-T', 'ek']
    for port in used_port:
        command.append("-d udp.port==" + port + ",rtp")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, encoding = 'utf-8', errors="ignore",stderr=None)
    try:
        output, error = process.communicate()
        return output
    except Exception as e:
        print ("Errore in pcap_to_json " + str(e))
        process.kill()
    return output


#Read json created with tshark and put it in a list
def json_to_list(output):
    
    def rtp_insert(obj, unique_flow, dict_flow_data, dictionary):
        
        # Retrive flow information 
        ssrc = obj['layers']['rtp']['rtp_rtp_ssrc']
        source_addr = obj['layers']['ip']['ip_ip_src']
        dest_addr = obj['layers']['ip']['ip_ip_dst']
        source_port = int(obj['layers']['udp']['udp_udp_srcport'])
        dest_port = int(obj['layers']['udp']['udp_udp_dstport'])
        p_type = int(obj['layers']['rtp']['rtp_rtp_p_type'])
        
        # Save ssrc if new
        unique_tuple = (ssrc, source_addr, dest_addr, source_port, dest_port, p_type)
        unique_flow.add(unique_tuple)

        # Retrive packet information
        timestamp = float(obj['layers']['frame']['frame_frame_time_epoch'])
        frame_num = int(obj['layers']['frame']['frame_frame_number'])       
        len_udp = int(obj['layers']['udp']['udp_udp_length'])
        len_ip = int(obj['layers']['ip']['ip_ip_len'])
        len_frame = int(obj['layers']['frame']['frame_frame_len'])
        rtp_timestamp = int(obj['layers']['rtp']['rtp_rtp_timestamp'])
        rtp_seq_num = int(obj['layers']['rtp']['rtp_rtp_seq'])
                
        # Add new packet to dictionary
#        columns = ['frame_num', 'p_type', 'len_udp', 'len_ip', 'len_frame', 'timestamps', 'rtp_timestamp', 'rtp_seq_num']
        data = [frame_num, p_type, len_udp, len_ip, len_frame, 
                timestamp, rtp_timestamp, rtp_seq_num]
        
        if unique_tuple in dictionary:
            dictionary[unique_tuple].append(data)
        else:
            dictionary[unique_tuple] = []
            dictionary[unique_tuple].append(data)
    

    l_rtp = []
    l_non_rtp = []
    l_stun = []
    l_rtcp = []
    l_turn = []
    l_tcp = []
    l_only_udp = []
    l_rtp_other = []
    l_dtls = []
    l_rtp_event = []
    l_mdns = []
    l_dns = []
    l_other = []
    
    dict_data = {}

    #Find RTP flows
    unique_flow = set()
    dict_flow_data = {}
    
    # df containign unique flow
    df_unique_flow= pd.DataFrame(columns = ['ssrc',
                           'source_addr',
                           'dest_addr',
                           'source_port',
                           'dest_port',
                           'rtp_p_type'])
    

    # Analyze each packet
    for obj in decode_stacked(output):
        #remove instances which have only index:date and type:pcap_file
        if 'index' in obj.keys():
            continue
        elif 'stun' in obj['layers'].keys():
            l_stun.append(obj)
        elif 'dns' in obj['layers'].keys():
            l_dns.append(obj)
        elif 'mdns' in obj['layers'].keys():
            l_mdns.append(obj)
        elif 'dtls' in obj['layers'].keys():
            l_dtls.append(obj)
        elif 'rtcp' in obj['layers'].keys():
            l_rtcp.append(obj)
        elif 'turn' in obj['layers'].keys():
            l_turn.append(obj)
        elif 'tcp' in obj['layers'].keys():
            l_tcp.append(obj)
        elif (('rtp' not in obj['layers'].keys()) & ('udp' in obj['layers'].keys())):
            l_only_udp.append(obj)
        elif ('rtpevent' in obj['layers'].keys()):
            l_rtp_event.append(obj)
        elif (('rtp' in obj['layers'].keys()) & ('rtpevent' not in obj['layers'].keys())):
            if (len(obj['layers']['rtp']) == 1):
                l_rtp_other.append(obj)
            else:
                rtp_insert(obj, unique_flow, dict_flow_data, dict_data)
        else:
            l_other.append(obj)
            
    for x in unique_flow:
        columns = ['frame_num', 'p_type', 'len_udp', 'len_ip', 'len_frame', 'timestamps', 'rtp_timestamp', 'rtp_seq_num']
        dict_flow_data[x] = pd.DataFrame(dict_data[x], columns=columns)
        df_unique_flow = df_unique_flow.append({
                'ssrc': x[0], 'source_addr': x[1],
                'dest_addr': x[2], 'source_port': x[3],
                'dest_port': x[4], 'rtp_p_type': x[5]}, ignore_index = True)
    #print("df_unique_flow shape: " + str(df_unique_flow.shape))
    #print("unique_flow shape: " + str(len(unique_flow)))
    #print("dictionaty shape: " + str(len(dict_flow_data)))
    return dict_flow_data, df_unique_flow, l_rtp, l_non_rtp, l_stun, l_rtcp, l_turn, l_tcp, l_only_udp, unique_flow





def calculate_packet_loss(dict_flow_data):
     
    #Calculate packet loss
    dict_flow_packet_loss = {}
    for flow_id in dict_flow_data:
        seq = dict_flow_data[flow_id]['rtp_seq_num'].sort_values()
        seq_diff = (seq - seq.shift()).fillna(1)
        dict_flow_packet_loss[flow_id] = (seq_diff.where(seq_diff != 1)-1).sum()
    
    print("Packet losses: ", dict_flow_packet_loss)
    
    return dict_flow_packet_loss  


def inter_statistic (dict_flow_data):
    global LEN_DROP
    for flow_id in dict_flow_data:
        
        dict_flow_data[flow_id]["interarrival"] = dict_flow_data[flow_id]["timestamps"].diff()
        dict_flow_data[flow_id]["rtp_interarrival"] = dict_flow_data[flow_id]["rtp_timestamp"].diff()
        dict_flow_data[flow_id]["interlength_udp"] = dict_flow_data[flow_id]["len_udp"].diff()
        dict_flow_data[flow_id]["label"] = 1 if dict_flow_data[flow_id]["len_udp"].mean() < 500 else 0 #1 SE AUDIO 0 SE VIDEO
        indexNames = dict_flow_data[flow_id][ dict_flow_data[flow_id]['interarrival'] > 1 ].index
        # Delete these row indexes from dataFrame
        LEN_DROP = LEN_DROP + len(indexNames)
        dict_flow_data[flow_id].drop(indexNames , inplace=True)
    return dict_flow_data


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


def merge_csv (directory):
    
    dataset_final = pd.DataFrame()
    for r, d, f in os.walk(directory):
        for file in f:
            if '.csv' in file:
                df_app = pd.read_csv(os.path.join(r, file))
                dataset_final = pd.concat([dataset_final, df_app])            
          
    dataset_final.to_csv( os.path.join(directory, "dataset.csv") )
    return dataset_final



#%%  
if __name__ == "__main__":   
    
    import argparse
    import os
    
#    parser = argparse.ArgumentParser()
#    parser.add_argument ("-d", "--directory", help = "Master directory", required = True)
#    parser.add_argument ("-j", "--join", help = "Join all .csv" , action='store_true')
#    parser.add_argument ("-p", "--plot", help = "Plot info" , action='store_true')
#    args = parser.parse_args()
#    directory_p = args.directory
    
    #If you want to do by hand
    directory_p = '/home/dena/Documents/Cisco_start/Captures/Try_classification'
    
    pcap_app = []
    for r, d, f in os.walk(directory_p):
        for file in f:
            if ('.pcap' in file or '.pcapng' in file):
                pcap_app.append(os.path.join(r, file)) 
    print(pcap_app)

    #For each .pcap in the folders, do the process
    for source_pcap in pcap_app:      
        pcap_path = os.path.dirname(source_pcap)
        output = pcap_to_json(source_pcap)
        dict_flow_data, df_unique_flow, l_rtp, l_non_rtp, l_stun, l_rtcp, l_turn, l_tcp, l_only_udp, unique_flow = \
            json_to_list(output)
        dict_flow_data = inter_statistic (dict_flow_data)
        
        #Make training dataframe
        df_train = pd.DataFrame()
        for flow_id in dict_flow_data.keys():
            
            datetime = pd.to_datetime(dict_flow_data[flow_id]["timestamps"], unit = 's').rename('times')
            dict_flow_data[flow_id].set_index(datetime, inplace = True)
            dict_flow_data[flow_id] = dict_flow_data[flow_id].dropna()
            train = dict_flow_data[flow_id].resample('s').agg(
                    {'interarrival' : ['std', 'mean', p25, p50, p75],
                     'len_udp' : ['std', 'mean', 'count', kbps, p25, p50, p75],
                     'interlength_udp' : ['mean', p25, p50, p75],
                     'rtp_interarrival' : ['std', 'mean', zeroes_count],
                     "label": [value_label]
                     })
            df_train = pd.concat([df_train, train])
        
        dataset_dropped = df_train.dropna()
        new_header = []
        for h in dataset_dropped.columns:
            new_header.append(h[0] + "_" + h[1])
        dataset_dropped.columns = dataset_dropped.columns.droplevel()
        dataset_dropped.columns = new_header
        dataset_dropped = dataset_dropped.rename(columns={'label_value_label': 'label'})
        dataset_dropped = dataset_dropped.rename(columns={'len_udp_kbps': 'kbps'})
        dataset_dropped = dataset_dropped.rename(columns={'len_udp_count': 'num_packets'})
        dataset_dropped = dataset_dropped.rename(columns={'rtp_interarrival_std': 'rtp_inter_timestamp_std'})
        dataset_dropped = dataset_dropped.rename(columns={'rtp_interarrival_mean': 'rtp_inter_timestamp_mean'})
        dataset_dropped = dataset_dropped.rename(columns={'rtp_interarrival_zeroes_count': 'rtp_inter_timestamp_num_zeros'})
        name = os.path.basename(source_pcap).split(".")[0]

        pcap_path = os.path.join(pcap_path, name)
        dataset_dropped.to_csv( pcap_path + ".csv" )
        
        html = df_unique_flow.to_html()
        with open(pcap_path + ".html", "w") as text_file:
            text_file.write(html)

        with open(pcap_path + "_info.txt", "w") as file:
            string = "Pacchetti droppati %s" % LEN_DROP
            file.write(string)
#%%
    if (args.join):
        merge_csv(directory_p)
    if (args.plot):
        plot_stuff(pcap_path, dict_flow_data, df_unique_flow)