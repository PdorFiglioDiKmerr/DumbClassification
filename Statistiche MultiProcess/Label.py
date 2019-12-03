import pandas as pd

def labelling (dict_flow_data, audio = None, video = None, ip = None):

    if ((audio is not None or video is not None) and ip is not None):
        for flow_id in dict_flow_data:
            print("hand-labeling")
            if ( (audio and ip) in flow_id ):
                dict_flow_data[flow_id]["label"] = 1 #audio
            elif ( (video and ip) in flow_id ):
                dict_flow_data[flow_id]["label"] = 0 # video
            else:
                print("Pay attenction: Labbelling on Meetings Capture is failed, auto-label is used")
                dict_flow_data[flow_id]["label"] = 1 if dict_flow_data[flow_id]["len_udp"].mean() < 500 else 0 #1 SE AUDIO 0 SE VIDEO

    else:
        for flow_id in dict_flow_data:
            dict_flow_data[flow_id]["label"] = 1 if dict_flow_data[flow_id]["len_udp"].mean() < 500 else 0 #1 SE AUDIO 0 SE VIDEO
            
    return dict_flow_data
