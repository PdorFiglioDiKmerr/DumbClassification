import pandas as pd

def inter_statistic (dict_flow_data, LEN_DROP):
    for flow_id in dict_flow_data:

        dict_flow_data[flow_id]["interarrival"] = dict_flow_data[flow_id]["timestamps"].diff()
        dict_flow_data[flow_id]["rtp_interarrival"] = dict_flow_data[flow_id]["rtp_timestamp"].diff()
        dict_flow_data[flow_id]["interlength_udp"] = dict_flow_data[flow_id]["len_udp"].diff()
        dict_flow_data[flow_id]["inter_time_sequence"] =  dict_flow_data[flow_id]["rtp_timestamp"] - dict_flow_data[flow_id]["rtp_seq_num"]
        indexNames = dict_flow_data[flow_id][ dict_flow_data[flow_id]['interarrival'] > 1 ].index
        # Delete these row indexes from dataFrame
        LEN_DROP = LEN_DROP + len(indexNames)
        dict_flow_data[flow_id].drop(indexNames , inplace=True)
    return dict_flow_data, LEN_DROP
