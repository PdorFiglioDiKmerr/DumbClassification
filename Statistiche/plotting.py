import pandas as pd
import matplotlib.pyplot as plt

def make_rtp_data(dict_flow_data):
    
    packets_per_second = {}
    kbps_series = {}
    inter_packet_gap_s = {}
    inter_rtp_timestamp_gap = {}
    
    for flow_id in dict_flow_data:
        
        inner_df = dict_flow_data[flow_id].sort_values('timestamps')
        inter_packet_gap_s[flow_id] = (inner_df.timestamps.diff()).dropna()
        inter_rtp_timestamp_gap[flow_id] = (inner_df.rtp_timestamp.diff()).dropna()
        
        # Need to define an index in order to use resample method
        datetime = pd.to_datetime(inner_df.timestamps, unit = 's')
        inner_df = inner_df.set_index(datetime)
        packets_per_second[flow_id] = inner_df.iloc[:,0].resample('S').count()
        kbps_series[flow_id] = inner_df['len_frame'].resample('S').sum()*8/1024

    return packets_per_second, kbps_series, inter_packet_gap_s, inter_rtp_timestamp_gap



def plot_stuff(pcap_path, dict_flow_df, df_unique):


    import matplotlib.dates as mdates
    from matplotlib import rcParams, rc

    font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 18}

    rc('font', **font)
    rcParams["figure.figsize"] = (16,9)
    rcParams['lines.linewidth'] = 1
    plt.ioff()


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
    fig, ax = plt.subplots(figsize = (16,12))
    for rtp_flow in dict_flow_df.keys():
        if rtp_flow[1].startswith('192.'):
            packets_per_second[rtp_flow].plot(lw = 3, label = rtp_flow, ax=ax)
        else:
            packets_per_second[rtp_flow].plot(lw = 3, linestyle='--', label = rtp_flow, ax=ax)
    leg = plt.legend(loc='lower left', bbox_to_anchor=(0.0, 1.1), ncol=2, fontsize=12)
    plt.grid(which = 'both')
    plt.title(t)
    plt.tight_layout()
    save_photo(pcap_path, t)
    
    
    #Plot Bitrate in time
    t = 'Bitrate in kbps'
    plt.figure(figsize = (18,12))
    for rtp_flow in dict_flow_df.keys():
        if rtp_flow[1].startswith('192.'):
            kbps_series[rtp_flow].plot(lw = 3, label = rtp_flow)
        else:
            kbps_series[rtp_flow].plot(lw = 3, linestyle='--', label = rtp_flow)
    plt.grid(which = 'both')
    plt.legend(loc='lower left', bbox_to_anchor=(0.0, 1.1), ncol=2, fontsize=12)
    plt.title(t)
    plt.tight_layout()
    save_photo(pcap_path, t)


    #Histogram of packet length
    for rtp_flow in dict_flow_df.keys():
        plt.figure()
        dict_flow_df[rtp_flow]['len_frame'].hist(label=rtp_flow, alpha=0.7, bins=50, color='#815EA4')

        t = 'Packet length histogram of ' + tuple_to_string(rtp_flow)
        plt.title(t)
        plt.ylabel('Occurences')
        plt.xlabel('Packet length in bytes')
        plt.tight_layout()
        save_photo(pcap_path, t, tuple_to_string(rtp_flow))


     #Packet length in time
    for rtp_flow in dict_flow_df.keys():
        plt.figure()
        dict_flow_df[rtp_flow]['len_frame'].plot(color='#815EA4')
        t = 'Packet length in time ' + tuple_to_string(rtp_flow)
        plt.title(t)
        plt.ylabel('Bytes')
        plt.tight_layout()
        plt.grid(which = 'both')
        save_photo(pcap_path, t, tuple_to_string(rtp_flow))


    #Inter-packet gap in time
    for rtp_flow in dict_flow_df.keys():
        plt.figure()
        if len(inter_packet_gap_s[rtp_flow]) != 0:
            inter_packet_gap_s[rtp_flow].plot(color='r')
            t = 'Inter packet gap in time ' + tuple_to_string(rtp_flow)
            plt.title(t)
            plt.xlabel('Seconds')
            plt.tight_layout()
            plt.grid(which = 'both')
            save_photo(pcap_path, t, tuple_to_string(rtp_flow))

    #Inter-packet gap histogram
    for rtp_flow in dict_flow_df.keys():
        plt.figure()
        if len(inter_packet_gap_s[rtp_flow]) != 0:
             inter_packet_gap_s[rtp_flow].hist(color='r', bins=50, alpha=0.7)
             t = 'Inter packet gap histogram ' + tuple_to_string(rtp_flow)
             plt.grid(b=True)
             plt.xlabel('Seconds')
             plt.ylabel('Occurences')
             plt.title(t)
             plt.tight_layout()
             save_photo(pcap_path, t, tuple_to_string(rtp_flow))



    m = 200
    for rtp_flow in dict_flow_df.keys():
         fig, ax = plt.subplots()
         dict_flow_df[rtp_flow]['rtp_timestamp'][:m].plot(style='co', ax=ax)
         t = 'First '+str(m)+'RTP timestamps in time ' + tuple_to_string(rtp_flow)
         plt.title(t, y=1.05)
         plt.ylabel('RTP timestamp units')
         plt.tight_layout()
         plt.grid(which = 'both')
         save_photo(pcap_path, t, tuple_to_string(rtp_flow))

     #Inter rtp timestamp gap histogram
    for rtp_flow in dict_flow_df.keys():
         plt.figure()
         if len(inter_rtp_timestamp_gap[rtp_flow]) != 0:
             inter_rtp_timestamp_gap[rtp_flow].hist(color='c', bins=50)
             t = 'Inter RTP timestamp in time ' + tuple_to_string(rtp_flow)
             plt.title(t)
             plt.xlabel('RTP timestamp units')
             plt.tight_layout()
             plt.grid(b=True)
             save_photo(pcap_path, t, tuple_to_string(rtp_flow))