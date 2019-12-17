import pandas as pd
import os
from Decode import decode_stacked


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
        try:
            rtp_csrc = obj['layers']['rtp']['rtp_csrc_items_rtp_csrc_item']
        except:
            rtp_csrc = "fec"
        # Add new packet to dictionary
#        columns = ['frame_num', 'p_type', 'len_udp', 'len_ip', 'len_frame', 'timestamps', 'rtp_timestamp', 'rtp_seq_num']
        data = [frame_num, p_type, len_udp, len_ip, len_frame,
                timestamp, rtp_timestamp, rtp_seq_num, rtp_csrc]

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
    l_error = []
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
        if 'ipv6' in obj['layers'].keys():
            continue

        if 'stun' in obj['layers'].keys():
            if 'stun_stun_channel' in obj['layers']["stun"]:
                l_turn.append(obj)
            else:
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
            if len(obj['layers']['rtp']) <= 3:
                l_rtp_other.append(obj)
            else:
                try:
                    l_rtp.append(obj)
                    rtp_insert(obj, unique_flow, dict_flow_data, dict_data)
                except:
                    l_error.append(obj)
                    
        else:
            l_other.append(obj)

    for x in unique_flow:
        columns = ['frame_num', 'p_type', 'len_udp', 'len_ip', 'len_frame', 'timestamps', 'rtp_timestamp', 'rtp_seq_num', 'rtp_csrc']
        dict_flow_data[x] = pd.DataFrame(dict_data[x], columns=columns)
        df_unique_flow = df_unique_flow.append({
                'ssrc': x[0], 'source_addr': x[1],
                'dest_addr': x[2], 'source_port': x[3],
                'dest_port': x[4], 'rtp_p_type': x[5]}, ignore_index = True)
    #print("df_unique_flow shape: " + str(df_unique_flow.shape))
    #print("unique_flow shape: " + str(len(unique_flow)))
    #print("dictionaty shape: " + str(len(dict_flow_data)))
    return dict_flow_data, df_unique_flow, l_rtp, l_non_rtp, l_stun, l_rtcp, l_turn, l_tcp, l_only_udp, unique_flow, l_error
