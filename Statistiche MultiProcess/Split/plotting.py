import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
#import matplotlib as mpl
#mpl.style.use('Pastel1')
matplotlib.rcParams.update({'font.size': 30})
plt.rcParams['font.size'] = 22
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 25
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['figure.titlesize'] = 15


def make_rtp_data(dict_flow_data):

    packets_per_second = {}
    kbps_series = {}
    inter_packet_gap_s = {}
    inter_rtp_timestamp_gap = {}

    for flow_id in dict_flow_data:
        #If the index is already datetime
        if isinstance(dict_flow_data[flow_id].index, pd.DatetimeIndex):
            inner_df = dict_flow_data[flow_id].sort_index().reset_index()
        else:
            inner_df = dict_flow_data[flow_id].sort_values('timestamps')

        #Find the timestamps column (should be only one with datetime)
#        timestamps_column = inner_df.select_dtypes(include=['datetime64']).iloc[:,0]
        inter_packet_gap_s[flow_id] = inner_df['interarrival'].dropna()
        inter_rtp_timestamp_gap[flow_id] = inner_df['rtp_interarrival'].dropna()

        # Need to define a datetime index to use resample
        datetime = pd.to_datetime(inner_df.timestamps, unit = 's')
        inner_df = inner_df.set_index(datetime)

        packets_per_second[flow_id] = inner_df.iloc[:,0].resample('S').count()
        kbps_series[flow_id] = inner_df['len_frame'].resample('S').sum()*8/1024
    return packets_per_second, kbps_series, inter_packet_gap_s, inter_rtp_timestamp_gap



def plot_stuff(pcap_path, dict_flow_df, df_unique):


    import matplotlib.dates as mdates
    from matplotlib import rcParams, rc
    rcParams["figure.figsize"] = (16,9)
    plt.ioff()
    class_dict = {0 : "Audio", 1 : "Video", 2 : "FEC Video",3: "Screen Sharing", 4 : "FEC Audio", 5 : "HQ", 6 : "LQ", 7 : "MQ"}


    def save_photo(pcap_path, t, flow=None):

        import os
        dpi = 100
        save_dir = os.path.join(pcap_path, "Plots")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if flow == None:
            plt.savefig(os.path.join(save_dir, t + '.png'), dpi=dpi)
        else:
            save_dir_flow = os.path.join(save_dir, flow)
            if not os.path.exists(save_dir_flow):
                os.makedirs(save_dir_flow)
            plt.savefig(os.path.join(save_dir_flow, t +'.png'), dpi = dpi)
        plt.close()

    #Convert tuple to string for naming purposes
    def tuple_to_string(tup):
        tup_string = ''
        for i in range(len(tup)):
            if i == len(tup)-1:
                tup_string += str(tup[i])
            else:
                tup_string += str(tup[i])+'_'
        tup_string = tup_string.replace('.','-')
        return tup_string


    packets_per_second, kbps_series, inter_packet_gap_s, inter_rtp_timestamp_gap = \
        make_rtp_data(dict_flow_df)

    #Plot stuff

    #Plot Packets per second in time
    t = 'Packets per second'
    plt.figure()
    #plt.figure(figsize = (16,12))
    for rtp_flow in sorted(dict_flow_df.keys()):
        if rtp_flow[1].startswith('192.'):
            plt.plot(packets_per_second[rtp_flow], linewidth = 2, label =  rtp_flow[0]+\
            " quality: "+ str(class_dict[dict_flow_df[rtp_flow]["label"][0]])+" sent")
        else:
            plt.plot(packets_per_second[rtp_flow], linewidth = 2, linestyle = "--", label = rtp_flow[0]+\
            " quality: "+ str(class_dict[dict_flow_df[rtp_flow]["label"][0]]) + " recived")

    leg = plt.legend(loc='lower left', bbox_to_anchor=(0.0, 1.1), ncol=2, fontsize=12)
    plt.grid(which = 'both')
    plt.title(t+ " quality: "+ str(class_dict[dict_flow_df[rtp_flow]["label"][0]]))
    plt.tight_layout()
    plt.xlabel("time")
    plt.ylabel("Packets/s")
    save_photo(pcap_path, t)
    #Histogram of Packets/s

    for rtp_flow in dict_flow_df.keys():
        plt.figure()
        if rtp_flow[1].startswith('192.'):
            sns.distplot(packets_per_second[rtp_flow], color="#650C10", label =  rtp_flow[0]+\
            " quality: "+ str(class_dict[dict_flow_df[rtp_flow]["label"][0]])+" sent", bins = 50, \
            hist_kws=dict(alpha=0.7))
        else:
            sns.distplot(packets_per_second[rtp_flow], color="#8B575C", label =  rtp_flow[0]+\
            " quality: "+ str(class_dict[dict_flow_df[rtp_flow]["label"][0]])+" recived", bins = 50,\
            hist_kws=dict(alpha=0.7))
        t = 'Packets per second density'
        plt.title(t +" quality: "+ str(class_dict[dict_flow_df[rtp_flow]["label"][0]]))
        plt.ylabel('Density (kde)')
        plt.xlabel('Packets per seconds ')
        plt.tight_layout()
        plt.grid (which = 'both')
        plt.legend()
        save_photo(pcap_path, t, tuple_to_string(rtp_flow))
    #Plot Bitrate in time

    t = 'Bitrate in kbps'
    plt.figure()
    for rtp_flow in sorted(dict_flow_df.keys()):
        if rtp_flow[1].startswith('192.'):
            plt.plot(kbps_series[rtp_flow], linewidth=2.5,  label = rtp_flow[0]+\
            " quality: "+ str(class_dict[dict_flow_df[rtp_flow]["label"][0]]) + " sent" )
        else:
            plt.plot(kbps_series[rtp_flow], linewidth = 2.5, linestyle = "--", label =  rtp_flow[0]+\
            " quality: "+ str(class_dict[dict_flow_df[rtp_flow]["label"][0]]) + " recived")
    plt.grid(which = 'both')
    plt.legend(loc='lower left', bbox_to_anchor=(0.0, 1.1), ncol=2, fontsize=12)
    plt.title(t )
    plt.xlabel("time")
    plt.ylabel("kbps")
    plt.tight_layout()
    save_photo(pcap_path, t)

    for rtp_flow in dict_flow_df.keys():
        plt.figure()
        if rtp_flow[1].startswith('192.'):
            sns.distplot(kbps_series[rtp_flow], color="#650C10", label =  rtp_flow[0]+\
            " quality: "+ str(class_dict[dict_flow_df[rtp_flow]["label"][0]])+" sent", bins = 50,\
            hist_kws=dict(alpha=0.7))
        else:
            sns.distplot(kbps_series[rtp_flow], color="#8B575C", label =  rtp_flow[0]+\
            " quality: "+ str(class_dict[dict_flow_df[rtp_flow]["label"][0]])+" recived", bins = 50,\
            hist_kws=dict(alpha=0.7))
        t = 'Bit rate density'
        plt.title(t +" quality: "+ str(class_dict[dict_flow_df[rtp_flow]["label"][0]]))
        plt.ylabel('Density (kde)')
        plt.xlabel('kbps ')
        plt.tight_layout()
        plt.grid (which = 'both')
        plt.legend()
        save_photo(pcap_path, t, tuple_to_string(rtp_flow))

    #Histogram of packet length
    for rtp_flow in dict_flow_df.keys():
        plt.figure()
        sns.distplot(dict_flow_df[rtp_flow]['len_frame'], color="#001427", label = rtp_flow, bins = 50,\
        hist_kws=dict(alpha=0.7))
        #dict_flow_df[rtp_flow]['len_frame'].hist(label=rtp_flow, alpha=0.7, bins=50, color='#815EA4')
        t = 'Packet length density'
        plt.title(t +" quality: "+ str(class_dict[dict_flow_df[rtp_flow]["label"][0]]))
        plt.ylabel('Density (kde)')
        plt.xlabel('Packet length in bytes')
        plt.tight_layout()
        plt.grid (which = 'both')
        plt.legend()
        save_photo(pcap_path, t, tuple_to_string(rtp_flow))


     #Packet length in time
    for rtp_flow in dict_flow_df.keys():
        plt.figure()
        plt.plot(dict_flow_df[rtp_flow]['len_frame'], "o", color='#815EA4', label = rtp_flow)
        t = 'Packet length in time'
        plt.title(t +" quality: "+ str(class_dict[dict_flow_df[rtp_flow]["label"][0]]))
        plt.ylabel('Bytes')
        plt.tight_layout()
        plt.grid(which = 'both')
        plt.legend()
        save_photo(pcap_path, t, tuple_to_string(rtp_flow))


    #Inter-packet gap in time
    for rtp_flow in dict_flow_df.keys():
        plt.figure()
        if len(inter_packet_gap_s[rtp_flow]) != 0:
            plt.plot(inter_packet_gap_s[rtp_flow],'ro', label = rtp_flow )
            t = 'Inter-arrival '
            plt.title(t +" quality: "+ str(class_dict[dict_flow_df[rtp_flow]["label"][0]]))
            plt.xlabel('Seconds')
            plt.tight_layout()
            plt.grid(which = 'both')
            plt.legend()
            save_photo(pcap_path, t, tuple_to_string(rtp_flow))

    #Inter-packet gap histogram
    for rtp_flow in dict_flow_df.keys():
        plt.figure()
        if len(inter_packet_gap_s[rtp_flow]) != 0:
            sns.distplot(inter_packet_gap_s[rtp_flow], color="#0F5678" , bins = 50, label = rtp_flow,\
            hist_kws=dict(alpha=0.7))
            #inter_packet_gap_s[rtp_flow].hist(color='r', bins=50, alpha=0.7)
            t = 'Inter-arrival density'
            plt.xlabel('Seconds')
            plt.ylabel('Density (kde)')
            plt.title(t +" quality: "+ str(class_dict[dict_flow_df[rtp_flow]["label"][0]]))
            plt.tight_layout()
            plt.grid(which='both')
            plt.legend()
            save_photo(pcap_path, t, tuple_to_string(rtp_flow))



    m = 100
    for rtp_flow in dict_flow_df.keys():
         plt.figure()
         plt.plot(dict_flow_df[rtp_flow]['rtp_timestamp'][:m], 'co', label = rtp_flow)
         t = 'First 100 RTP timestamps in time '
         plt.title(t +" quality: "+ str(class_dict[dict_flow_df[rtp_flow]["label"][0]]), y=1.05)
         plt.ylabel('RTP timestamp units')
         plt.tight_layout()
         plt.grid(which = 'both')
         plt.legend()
         save_photo(pcap_path, t, tuple_to_string(rtp_flow))

     #Inter rtp timestamp gap histogram
    for rtp_flow in dict_flow_df.keys():
         plt.figure()
         if len(inter_rtp_timestamp_gap[rtp_flow]) != 0:
             sns.distplot(inter_rtp_timestamp_gap[rtp_flow], color="#998D7D", bins = 50, label = rtp_flow,\
             hist_kws=dict(alpha=0.7))
             #inter_rtp_timestamp_gap[rtp_flow].hist(color='c', bins=50)
             t = 'Inter RTP timestamp in time density'
             plt.title(t + " quality: "+ str(class_dict[dict_flow_df[rtp_flow]["label"][0]]))
             plt.xlabel('RTP timestamp units')
             plt.ylabel('Density (kde)')
             plt.tight_layout()
             plt.grid(which='both')
             plt.legend()
             save_photo(pcap_path, t, tuple_to_string(rtp_flow))

    # for rtp_flow in dict_flow_df.keys():
    #     plt.figure()
    #     dict_flow_df[rtp_flow]["rtp_timestamp"].value_count().plot(kind = "bar")
    #     t = ' RTP_occurences' + tuple_to_string(rtp_flow)
    #     plt.title(t)
    #     plt.xlabel('RTP timestamp equal')
    #     plt.tight_layout()
    #     plt.grid(b=True)
    #     save_photo(pcap_path, t, tuple_to_string(rtp_flow))
