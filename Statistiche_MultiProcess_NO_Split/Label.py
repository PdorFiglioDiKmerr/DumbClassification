import pandas as pd


#ETICHETTATURA NOSTRA


def audio_vs_fec(dict_flow_data, flow_id):
    if (dict_flow_data[flow_id]["rtp_timestamp"] == 0).all():
        dict_flow_data[flow_id]["label"] =  4#FEC audio
        #print("FEC_0: {}".format(flow_id))
    elif check_fec_equal(dict_flow_data,flow_id):
        #print("FEC_same: {}".format(flow_id))
        dict_flow_data[flow_id]["label"] =  4#FEC audio ricevuto
    else:
        dict_flow_data[flow_id]["label"] =  0#audio
    return dict_flow_data[flow_id]

def video_vs_fec(dict_flow_data, flow_id, quality= None):
    if (dict_flow_data[flow_id]["rtp_timestamp"] == 0).all():
        dict_flow_data[flow_id]["label"] = 2 #FEC VIDEO
    elif check_fec_equal(dict_flow_data,flow_id):
        dict_flow_data[flow_id]["label"] = 2 #FEC VIDEO RICEVUTO
    elif (quality == "HQ"):
        dict_flow_data[flow_id]["label"] =  5#Video HQ
    elif (quality == "LQ"): #standard quality
        dict_flow_data[flow_id]["label"] =  6#Video LQ
    else:
        dict_flow_data[flow_id]["label"] =  1 #Video Root-Class
    return dict_flow_data[flow_id]

def automate_classify (dict_flow_data, flow_id):
    if dict_flow_data[flow_id]["len_udp"].mean() < 400:
        dict_flow_data[flow_id] = audio_vs_fec(dict_flow_data, flow_id)
    else:
        dict_flow_data[flow_id] = video_vs_fec(dict_flow_data, flow_id)
    return dict_flow_data[flow_id]


def labelling (dict_flow_data, audio = None, video = None, ip = None, screen = None, quality = None):

    if ((audio is not None or video is not None) and ip is not None):
        for flow_id in dict_flow_data:
            print("hand-labeling")
            if ( audio in flow_id and ip in flow_id ):
                dict_flow_data[flow_id] = audio_vs_fec(dict_flow_data, flow_id)
            elif ( video in flow_id and ip in flow_id ):
                dict_flow_data[flow_id] = video_vs_fec(dict_flow_data, flow_id, quality)
            elif (screen):
                dict_flow_data[flow_id]["label"] = 3 #Screen Sharing
            else:
                print("Pay attenction: Labbelling on Meetings Capture is failed, auto-label is used")
                dict_flow_data[flow_id] = automate_classify(dict_flow_data, flow_id)
    else:
        for flow_id in dict_flow_data:
            dict_flow_data[flow_id] = automate_classify(dict_flow_data, flow_id)
    return dict_flow_data



#CISCO ETICHETTATURA
def check_fec_equal (dict_flow_data, flow_id):
    try:
        cost = pow(2,16)
        ts = dict_flow_data[flow_id]["rtp_timestamp"]
        sn = dict_flow_data[flow_id]["rtp_seq_num"]
        ts_final = ts%cost
        return ts_final.equals(sn)
    except Exception as e:
        return False

def labelling2 (dict_flow_data):
    for flow_id in dict_flow_data:
        if not (any(dict_flow_data[flow_id]["rtp_csrc"].isin(["fec"]))):
            dict_flow_data[flow_id]["rtp_csrc"] = [ str( bin(int(x, 16))[2:] ) for x in dict_flow_data[flow_id]["rtp_csrc"] ]
            if (dict_flow_data[flow_id]["rtp_timestamp"] == 0).all(): #FEC Inviati
                dict_flow_data[flow_id] = fec_audio_video_csrc(dict_flow_data, flow_id)
            elif check_fec_equal(dict_flow_data,flow_id):
                dict_flow_data[flow_id] = fec_audio_video_csrc(dict_flow_data, flow_id)
            else:
                dict_flow_data[flow_id]["label2"] = [ 1 if x[-1] == str(1) else 0 for x in dict_flow_data[flow_id]["rtp_csrc"] ]
        else: #se compare una voce fec etichetta euristica
            dict_flow_data[flow_id] = fec_audio_video_euristic(dict_flow_data, flow_id)

    return dict_flow_data



def fec_audio_video_csrc(dict_flow_data, flow_id):

    dict_flow_data[flow_id]["label2"] = [ 2 if x[-1] == str(1) else 4 for x in dict_flow_data[flow_id]["rtp_csrc"] ]
    return dict_flow_data[flow_id]


def fec_audio_video_euristic(dict_flow_data, flow_id):


    if dict_flow_data[flow_id]["len_udp"].mean() < 400:
        dict_flow_data[flow_id]["label2"] =  4#FEC audio
    else:
        dict_flow_data[flow_id]["label2"] = 2 #FEC VIDEO
    return dict_flow_data[flow_id]
