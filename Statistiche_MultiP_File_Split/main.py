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
# class NoDaemonProcess(multiprocessing.Process):
#     # make 'daemon' attribute always return False
#     def _get_daemon(self):
#         return False
#     def _set_daemon(self, value):
#         pass
#     daemon = property(_get_daemon, _set_daemon)
#
# # We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# # because the latter is only a wrapper function, not a proper class.
# class MyPool(multiprocessing.pool.Pool):
#     Process = NoDaemonProcess
# #%%
# counter_ssrc = 0

#%%
n_process = 16

def split_file(source_pcap):

    num_packets = 300000
    #print ("Processo {}".format(source_pcap))
    name = os.path.basename(source_pcap).split(".")[0]
    pcap_path = os.path.dirname(source_pcap)
    new_dir = pcap_split (num_packets,source_pcap, pcap_path, name)
    new_dir_name = [os.path.join(new_dir,fs) for fs in os.listdir(new_dir)]
    result_list.append(new_dir_name)

def main2(new_dir_name_file, result_list):
        LEN_DROP = 0
        used_port = pcap_to_port(new_dir_name_file)
        result_list.append(used_port)
        #print("pcap2port terminati porte: {}".format(used_port))


if __name__ == "__main__":
    
    #logging.basicConfig(filename='example.log',filemode='w', level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument ("-d", "--directory", help = "Master directory", required = True)
    parser.add_argument ("-j", "--join", help = "Join all .csv" , action='store_true')
    parser.add_argument ("-p", "--plot", help = "Plot info" , action='store_true')
    args = parser.parse_args()
    directory_p = args.directory
    #If you want to do by hand
    #directory_p = r'C:\Users\Gianl\Desktop\Call_with_Chiara'

    pcap_app = []
    for r, d, f in os.walk(directory_p):
        for file in f:
            if ('.pcap' in file or '.pcapng' in file):
                pcap_app.append(os.path.join(r, file))
    #print("Pcap found: {}\n".format(pcap_app))
    #For each .pcap in the folders, do the process
    manager = multiprocessing.Manager()
    result_list = manager.list()
    #Splitto i pcap
    pool= multiprocessing.Pool(processes = n_process) #Limito il numero di processi ai core della cpu -1
    pool.map(split_file, pcap_app)
    pool.close()
    pool.join()
    
    list_app_name = []
    list_app_name = [j for i in result_list for j in i]
    #print ("list_app_name: {}\n".format(list_app_name))
    #logging.info("Finish Process split_file\n")
    #result_list = manager.list()
    result_list[:] = []

    #Cerco le porte
    pool= multiprocessing.Pool(processes = n_process) #Limito il numero di processi ai core della cpu -1
    pool_tuple = [(x, result_list) for x in list_app_name]
    pool.starmap(main2, pool_tuple)
    pool.close()
    pool.join()
    #logging.info("Finish Process main2\n")
    list_app_port = []
    list_app_port = [j for i in result_list for j in i]
    port_used = set(list(map(int, list_app_port)))
    result_list[:] = []
    #print(result_list)
    #Decodifico su porto e credo .csv
    pool= multiprocessing.Pool(processes = n_process) #Limito il numero di processi ai core della cpu -1
    pool_tuple = [(x, port_used) for x in list_app_name]  #result_list, args.plot
    pool.map(pcap_to_json, pool_tuple)
    pool.close()
    pool.join()
    #logging.info('Finished')

#%%
    if (args.join):
        merge_csv(directory_p)
