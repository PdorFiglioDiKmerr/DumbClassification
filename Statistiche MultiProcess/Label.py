import pandas as pd

def labelling (dict_flow_data, audio = None, video = None, ip = None, screen = None):

    if ((audio is not None or video is not None) and ip is not None):
        for flow_id in dict_flow_data:
            print("hand-labeling")
            if ( (audio and ip) in flow_id ):
                if (dict_flow_data[flow_id]["timestamps"] == 0).all():
                    dict_flow_data[flow_id]["label"] = 0 #FEC audio
                else:
                    dict_flow_data[flow_id]["label"] =  1#audio
            elif ( (video and ip) in flow_id ):
                if (dict_flow_data[flow_id]["timestamps"] == 0).all():
                    dict_flow_data[flow_id]["label"] = 2 #FEC VIDEO
                else:
                    dict_flow_data[flow_id]["label"] =  3#Video
            elif (screen):
                dict_flow_data[flow_id]["label"] = 4 #Screen Sharing
            else:
                print("Pay attenction: Labbelling on Meetings Capture is failed, auto-label is used")
                if dict_flow_data[flow_id]["len_udp"].mean() < 500:
                    if (dict_flow_data[flow_id]["timestamps"] == 0).all():
                        dict_flow_data[flow_id]["label"] = 0 #FEC audio
                    else:
                        dict_flow_data[flow_id]["label"] =  1#audio
                else:
                    if (dict_flow_data[flow_id]["timestamps"] == 0).all():
                        dict_flow_data[flow_id]["label"] = 2 #FEC Video
                    else:
                        dict_flow_data[flow_id]["label"] =  3#video
    else:
        for flow_id in dict_flow_data:
            if dict_flow_data[flow_id]["len_udp"].mean() < 500:
                if (dict_flow_data[flow_id]["timestamps"] == 0).all():
                    dict_flow_data[flow_id]["label"] = 0 #FEC audio
                else:
                    dict_flow_data[flow_id]["label"] =  1#audio
            else:
                if (dict_flow_data[flow_id]["timestamps"] == 0).all():
                    dict_flow_data[flow_id]["label"] = 2 #FEC video
                else:
                    dict_flow_data[flow_id]["label"] =  3#video
            
    return dict_flow_data
