#Open Tshark and run command to turn pcap to json
import subprocess
import pandas as pd
from Decode import decode_stacked
import os
import json
import sys
from Json2List import json_to_list
from json2stat import json2stat
import time
def pcap_to_port(source_pcap):
    try:
    # Retrive all STUN packets
        command = ['tshark', '-r', source_pcap, '-l', '-n', '-T', 'ek', '-Y (stun)']
        process = subprocess.run(command, stdout=subprocess.PIPE, encoding = 'utf-8', shell=False )
        output = process.stdout
    except Exception as e:
        print ("Errore in pcap_to_json: {}".format(e))
        process.kill()
        raise e
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
            continue
    return list(used_port)

def pcap_to_json(tuple_param): #source_pcap, used_port
    source_pcap = tuple_param[0]
    used_port = tuple_param[1]
    #print ("{} : {}".format(os.getpid(),tuple_param))
    name = os.path.basename(source_pcap).split(".")[0]
    pcap_path = os.path.dirname(source_pcap)
    json_path = os.path.join(pcap_path,name+".json")
    command = ['tshark', '-r', source_pcap, '-l', '-n', '-T', 'ek']
    for port in used_port:
        command.append("-d udp.port==" + str(port) + ",rtp")
    with open(os.path.join(pcap_path,name+".txt") ,"w",encoding = 'utf-8') as file:
        subprocess.Popen(command,stdout = file, stderr = None ,encoding = 'utf-8').communicate()
    with open(os.path.join(pcap_path,name+".txt") ,"r") as file:
        output = file.read()
    dict_flow_data, df_unique_flow, l_rtp, l_non_rtp, l_stun, l_rtcp, l_turn, l_tcp, l_only_udp, unique_flow, l_err = \
        json_to_list(output, json_path)
    print ("Json: {}  PID: {}".format(json_path, os.getpid()))
    json2stat(dict_flow_data, pcap_path, name)
    return 
