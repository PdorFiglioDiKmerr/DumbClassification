#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import json
import matplotlib.pyplot as plt
import sys
import numpy as np
import re
from plotting import plot_stuff
from MergeCSV import merge_csv
from InterStatistics import inter_statistic
from PacketLoss import calculate_packet_loss
from Json2List import json_to_list
from Pcap2Json import pcap_to_json, pcap_to_port
from plotting import make_rtp_data
from Label import labelling
from Label import labelling2
from split_pcap import pcap_split
import argparse
import os
import multiprocessing
import multiprocessing.pool
import time
import tqdm
from random import randint
import logging
#%%


#%%
if __name__ == "__main__":
    # name field :layers - rtp -  rtp_csrc_items_rtp_csrc_item
    # name filed: rtp_timestamp
  
    #If you want to do by hand
    source_pcap = r'C:\Users\Gianl\Desktop\Test_new_plot_SS\Call_1_5_min_SS.pcapng'
    used_port = pcap_to_port(source_pcap)
    #PREPARO TUPLA PER PCAP_TO_JSON 
    #terzo parametro se True il video è classficato screen sharing
    #quarto parametro è la qualità ["HQ", "MQ", "LQ", None]
    #quinto parametro se True fa i plot
    info = (source_pcap, used_port, True, None, False)
    pcap_to_json(info)
    