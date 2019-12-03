import os
import pandas as pd
from json import JSONDecoder, JSONDecodeError
import sys
import re
import warnings
import numpy as np
warnings.filterwarnings("ignore")

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
def pcap_to_json(source_pcap, protocol_policy):
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
        if 'index' in obj.keys():
            continue
        if 'stun' in obj['layers'].keys() and 'stun_stun_type' in obj['layers']["stun"] and "0x00000101" in obj['layers']["stun"]["stun_stun_type"]:          #0x0101 means success
            used_port.add(obj['layers']["udp"]["udp_udp_srcport"])
            used_port.add(obj['layers']["udp"]["udp_udp_dstport"])
            
#    print("Check on Wireshark if there are other UDP flow not recognized: ")
#    for i in used_port:
#        sys.stdout.write("udp.port == "+ str(i) +" || ")
    
    command = ['tshark', '-r', source_pcap, '-l', '-n', '-T', 'ek', protocol_policy, 'rtp']
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


#def compute_entropy(hex_string):
#    from math import e, log
#    bin_string = bin(int(hex_string, 16))[2:].zfill(len(hex_string)*4)
#    bin_string = list(bin_string)
#    n_labels = len(bin_string)
#    if n_labels <= 1:
#        return 0
#    value, counts = np.unique(bin_string, return_counts=True)
#    probs = counts / n_labels
#    n_classes = np.count_nonzero(probs)
#    if n_classes <= 1:
#        return 0
#    base = e
#    ent = 0
#    for i in probs:
#        ent -= i * log(i, base)
#    return ent

#Read json created with tshark and put it in a list
def json_to_list(output):
    
    def multicast_address(obj):
        # Multicast MAC
        if 'ff:ff:ff:ff:ff:ff' in obj['layers']['eth']['eth_eth_dst']:
            return True
        
        if '33:33' in obj['layers']['eth']['eth_eth_dst'][0:5]:
            return True
        
        # Multicast IPv4
        if 'ip' in obj['layers'].keys() and '.' not in obj['layers']['ip']['ip_ip_dst'][0:3]:
            bin_str = "{0:b}".format(int(obj['layers']['ip']['ip_ip_dst'][0:3]))
            if bin_str.startswith("1110"):
                return True
        
        # Multicast IPv6
        if 'ipv6' in obj['layers'].keys() and obj['layers']['ipv6']['ipv6_ipv6_dst'][0:4].startswith("ff00"):
            return True
        return False
        
        
    def udp_insert(obj, unique_flow, dict_flow_data, dictionary, payload_type_used):
        
        if 'stun' in obj['layers'].keys() or 'rtcp' in obj['layers'].keys():
            return
            
        # Retrive flow information
        if "ipv6" in obj['layers']:
            source_addr = obj['layers']['ipv6']['ipv6_ipv6_src']
            dest_addr = obj['layers']['ipv6']['ipv6_ipv6_dst']
        elif "ip" in obj["layers"]:
            source_addr = obj['layers']['ip']['ip_ip_src']
            dest_addr = obj['layers']['ip']['ip_ip_dst']
        source_port = int(obj['layers']['udp']['udp_udp_srcport'])
        dest_port = int(obj['layers']['udp']['udp_udp_dstport'])
        
        # Save ssrc if new
        unique_tuple = (source_addr, dest_addr, source_port, dest_port)
        unique_flow.add(unique_tuple)

        # Retrive packet information
        frame_num = int(obj['layers']['frame']['frame_frame_number'])       
        len_udp = int(obj['layers']['udp']['udp_udp_length'])
        len_frame = int(obj['layers']['frame']['frame_frame_len'])
        timestamp = float(obj['layers']['frame']['frame_frame_time_epoch'])
        
        label = "Not RTP"
        if 'rtp' in obj['layers'].keys() and 'rtp_rtp_version' in obj['layers']['rtp'] and '2' in obj['layers']['rtp']['rtp_rtp_version']:
            if 'rtp_rtp_p_type' in obj['layers']['rtp'] and obj['layers']['rtp']['rtp_rtp_p_type'] in payload_type_used:
                if source_port > 1023 and dest_port > 1023:
                    label = "RTP"
        
        if 'rtcp' in obj['layers'].keys():
            label = "RTP"
        
        if 'stun' in obj['layers'].keys():
            label = "RTP"
        
        
        data = [frame_num, len_udp, len_frame, timestamp, label]        
        if unique_tuple in dictionary:
            dictionary[unique_tuple].append(data)
        else:
            dictionary[unique_tuple] = []
            dictionary[unique_tuple].append(data)
    
    dict_data = {}
    payload_type_used = list(range(1, 20)) + list(range(25, 27)) + [28] + list(range(31, 35)) + list(range(96, 128))
    payload_type_used = [str(i) for i in payload_type_used]

    #Find RTP flows
    unique_flow = set()
    dict_flow_data = {}
    
    # df containign unique flow
    columns = ['source_addr','dest_addr','source_port','dest_port']
    df_unique_flow = pd.DataFrame(columns=columns)
    
    # Analyze each packet
    for obj in decode_stacked(output):
        if 'index' in obj.keys():
            continue
        if multicast_address(obj):
            continue
        if 'udp' in obj['layers'].keys():
            udp_insert(obj, unique_flow, dict_flow_data, dict_data, payload_type_used)

    for x in unique_flow:
        columns = ['frame_num', 'len_udp', 'len_frame', 'timestamps', 'label']
        dict_flow_data[x] = pd.DataFrame(dict_data[x], columns=columns)
        df_unique_flow = df_unique_flow.append({'source_addr': x[0], 'dest_addr': x[1], 
                'source_port': x[2], 'dest_port': x[3]}, ignore_index = True)

    return dict_flow_data, df_unique_flow, unique_flow

#Create nicknames for every flow
def make_nicknames(dict_flow_data):
    
    #New: packet lenght mean -> audio/video
    dict_flow_nickname = {}
    threshold = 300
    for key in dict_flow_data:
        mean = dict_flow_data[key]["len_frame"].mean()
        if mean > threshold:
            dict_flow_nickname[key] = "video"
        else:
            dict_flow_nickname[key] = "audio"
    return dict_flow_nickname

def calculate_packet_loss(dict_flow_data):
     
    #Calculate packet loss
    dict_flow_packet_loss = {}
    for flow_id in dict_flow_data:
        seq = dict_flow_data[flow_id]['rtp_seq_num'].sort_values()
        seq_diff = (seq - seq.shift()).fillna(1)
        dict_flow_packet_loss[flow_id] = (seq_diff.where(seq_diff != 1)-1).sum()
        
    return dict_flow_packet_loss

def inter_statistics(dict_flow_data):
    for flow_id in dict_flow_data:
        dict_flow_data[flow_id]["interarrival"] = dict_flow_data[flow_id]["timestamps"].diff()
        dict_flow_data[flow_id]["interlength_udp"] = dict_flow_data[flow_id]["len_udp"].diff()
        dict_flow_data[flow_id] = dict_flow_data[flow_id][dict_flow_data[flow_id].interarrival < 1]
    return dict_flow_data

def kbps(series):
    return series.sum()*8/1024

def max_count(series):
    a = len(series[series == "RTP"]) > len(series[series == "Not RTP"])
    return "RTP" if a else "Not RTP"

'''
def find_associated_flow(dict_flow_data, flow):
    associated_flow = flow[1], flow[0], flow[3], flow[2]
    if associated_flow in dict_flow_data:
        size_first_flow = len(dict_flow_data[flow])
        size_second_flow = len(dict_flow_data[associated_flow])
        ratio = size_first_flow/(size_first_flow+size_second_flow) if (size_first_flow+size_second_flow != 0) else 1
    else:
        ratio = 1
    dict_flow_data[flow]['ratio'] = ratio
'''    

def find_associated_flow(dataset_dropped):
    dataset_dropped = dataset_dropped.sort_index()
    ratio_columns = []
    for row in dataset_dropped.itertuples():
        flow = getattr(row, 'flow')
        associated_flow = flow[1], flow[0], flow[3], flow[2]
        start_timestamp = getattr(row, 'Index')
        end_timestamp = start_timestamp + pd.Timedelta(seconds=10)
        timestamp_range = pd.date_range(start=start_timestamp, end=end_timestamp, freq='1s')
        time_window = [x in timestamp_range for x in dataset_dropped.index]
        # Compute size(flow)
        flows = [x == flow for x in dataset_dropped.flow]
        indicesToKeep = time_window and flows
        df_same_flow = dataset_dropped.loc[indicesToKeep]
        size_flow = np.sum(df_same_flow.len_udp_packet_count, axis=0)
        # Compute size(reverse_flow)
        associated_flows = [x == associated_flow for x in dataset_dropped.flow]
        indicesToKeep = time_window and associated_flows
        df_reverse_flow = dataset_dropped.loc[indicesToKeep]
        size_reverse_flow = np.sum(df_reverse_flow.len_udp_packet_count, axis=0)
        # Compute ratio
        if size_flow != 0 or size_reverse_flow != 0:
            ratio = size_flow/(size_flow+size_reverse_flow)
        else:
            ratio = 1
        ratio_columns.append(ratio)
    dataset_dropped['ratio'] = ratio_columns
    return dataset_dropped

def percentile(n):
    def percentile_(x):
        return x.quantile(n*0.01)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

def packet_count(series):
    return len(series)

class pcap_manager():
    
    def __init__(self, time_window_size):
        self.separator = '/' if sys.platform.startswith("linux") else '\\'
        self.time_window_size = time_window_size
        self.seconds = int(time_window_size.split('s')[0])
    
    def pcap_to_df(self, source):
        filename = source.rsplit(self.separator, 1)[1]
        protocol_policy = "--disable-protocol" if 'game' in filename else '--enable-protocol'
        output = pcap_to_json(source, protocol_policy)
        dict_flow_data, df_unique_flow, unique_flow = json_to_list(output)
        dict_flow_data = inter_statistics(dict_flow_data) 
        dataset = pd.DataFrame()
        for flow_id in dict_flow_data:
            dict_flow_data[flow_id]["timestamps"] = pd.to_datetime(dict_flow_data[flow_id]["timestamps"], unit = 's')
            dict_flow_data[flow_id].set_index('timestamps', inplace = True)
            if len(dict_flow_data[flow_id]) > 0:
                #train = dict_flow_data[flow_id].resample(self.time_window_size).agg({
                #    'interarrival' : ['std', 'mean'], 'len_udp' : ['std', 'mean', packet_count, kbps],'interlength_udp' : ['mean'],
                #    'ratio': ['mean'], "label": max_count})
                train = dict_flow_data[flow_id].resample(self.time_window_size).agg({'interarrival' : ['std', 'mean',percentile(25),
                percentile(50), percentile(75)], 'len_udp' : ['std', 'mean', packet_count, 
                kbps, percentile(25), percentile(50), percentile(75)],'interlength_udp' : ['mean', percentile(25), percentile(50),
                percentile(75)], "label": max_count})
                train['flow'] = [flow_id] * len(train)
                train['len_udp']['kbps'] = train['len_udp']['kbps']/self.seconds
                train['len_udp']['packet_count'] = train['len_udp']['packet_count']/self.seconds
                dataset = pd.concat([dataset, train])
        dataset_dropped = dataset.dropna()
        
        # Remove double header
        new_header = []
        for i in dataset_dropped.columns:
            new_header.append(i[0] + "_" + i[1]) if i[1] != '' else new_header.append(i[0])
        dataset_dropped.columns = dataset_dropped.columns.droplevel()
        dataset_dropped.columns = new_header
        dataset_dropped = dataset_dropped.rename(columns={'label_max_count': 'label'})
        dataset_dropped = find_associated_flow(dataset_dropped) # Create 'ratio' column
        dataset_dropped = dataset_dropped.drop(columns = 'flow')
        
        # Put label as last column
        cols = dataset_dropped.columns.tolist()
        cols = cols[:-2] + cols[-1:] + cols[-2:-1]
        dataset_dropped = dataset_dropped[cols]
        return dataset_dropped
        
    def pcap_to_csv(self, source):
        dest = source.rsplit(self.separator, 1)[0]
        df_dataset = self.pcap_to_df(source)
        df_dataset.to_csv(dest + self.separator + "dataset.csv", sep=',', encoding='utf-8', index=False)
        
    def merge_pcap(self, directory):
        import concurrent.futures
        directory = directory[:-1] if directory[len(directory)-1] == self.separator else directory
        arr = os.listdir(directory)
        arr = [file for file in arr if '.' in file and file.split(".")[1].startswith(("pcapng", "pcap"))]
        arr = [directory + self.separator + file for file in arr]
        df_merged = pd.DataFrame()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(self.pcap_to_df, arr)            
        for df in results:
            df_merged = pd.concat([df_merged, df], ignore_index=True)
        df_merged.to_csv(directory + self.separator + "dataset.csv", sep=',', encoding='utf-8', index=False)