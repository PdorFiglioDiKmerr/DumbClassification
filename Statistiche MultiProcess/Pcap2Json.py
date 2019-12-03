#Open Tshark and run command to turn pcap to json
import subprocess
import pandas as pd
from Decode import decode_stacked

def pcap_to_json(source_pcap):
    import subprocess

    # Retrive all STUN packets
    command = ['tshark', '-r', source_pcap, '-l', '-n', '-T', 'ek', '-Y (stun)']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=None, encoding = 'utf-8', shell=False )
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
    process = subprocess.Popen(command, stdout=subprocess.PIPE, encoding = 'utf-8', errors="ignore",stderr=None, shell=False)
    try:
        output, error = process.communicate()
        return output
    except Exception as e:
        print ("Errore in pcap_to_json " + str(e))
        process.kill()
    return output
