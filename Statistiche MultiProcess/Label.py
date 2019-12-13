import pandas as pd

def audio_vs_fec(dict_flow_data, flow_id):
    if (dict_flow_data[flow_id]["rtp_timestamp"] == 0).all():
        dict_flow_data[flow_id]["label"] =  4#FEC audio
    else:
        dict_flow_data[flow_id]["label"] =  0#audio
    return dict_flow_data[flow_id]

def video_vs_fec(dict_flow_data, flow_id, quality= None):
    if (dict_flow_data[flow_id]["rtp_timestamp"] == 0).all():
        dict_flow_data[flow_id]["label"] = 2 #FEC VIDEO
    elif (quality == "HQ"):
        dict_flow_data[flow_id]["label"] =  1#Video HQ
    elif (quality == "SQ"): #standard quality
        dict_flow_data[flow_id]["label"] =  3#Video SQ
    else:
        dict_flow_data[flow_id]["label"] =  1
    return dict_flow_data[flow_id]

def automate_classify (dict_flow_data, flow_id):
    if dict_flow_data[flow_id]["len_udp"].mean() < 500:
        dict_flow_data[flow_id] = audio_vs_fec(dict_flow_data, flow_id)
    else:
        dict_flow_data[flow_id] = video_vs_fec(dict_flow_data, flow_id)
    return dict_flow_data[flow_id]

def labelling (dict_flow_data, audio = None, video = None, ip = None, screen = None):

    if ((audio is not None or video is not None) and ip is not None):
        for flow_id in dict_flow_data:
            print("hand-labeling")
            if ( audio in flow_id and ip in flow_id ):
                dict_flow_data[flow_id] = audio_vs_fec(dict_flow_data, flow_id)
            elif ( video in flow_id and ip in flow_id ):
                dict_flow_data[flow_id] = video_vs_fec(dict_flow_data, flow_id)
            elif (screen):
                dict_flow_data[flow_id]["label"] = 5 #Screen Sharing
            else:
                print("Pay attenction: Labbelling on Meetings Capture is failed, auto-label is used")
                dict_flow_data[flow_id] = automate_classify(dict_flow_data, flow_id)
    else:
        for flow_id in dict_flow_data:
            dict_flow_data[flow_id] = automate_classify(dict_flow_data, flow_id)
    return dict_flow_data


def labelling2 (dict_flow_data):
    for flow_id in dict_flow_data:
        dict_flow_data[flow_id]["rtp_csrc"] = [bin(int(x, 16))[2:] for x in dict_flow_data[flow_id]["rtp_csrc"]]
        dict_flow_data[flow_id]["label2"] = [1 if x[-1] == 1 else 0 for x in dict_flow_data[flow_id]["rtp_csrc"]]
    return dict_flow_data