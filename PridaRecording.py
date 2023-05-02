from open_ephys import analysis


import sys
import argparse

import pandas as pd
pd.set_option('display.max_rows', 40)
import numpy as np
import matplotlib.pyplot as plt 

import os 
import time
import ast
import bisect

class PridaRecording():
    '''
    Class for loading and analyzing Prida Lab recordings
    '''

    class Ripples:
        
        def __init__(self, base_directory):
            #TODO: Load ripples from events text file
            #self.ripples = pd.read_csv(os.path.join(base_directory, 'events', 'event_selected_manually.txt'), sep=' ', header=None)
            pass

    class Continuous:

        def __init__(self, recording, base_directory):

            self.metadata = {}

            self.metadata['sample_rate'] = recording.sample_rate

            data_path = base_directory

            dat_file = os.path.join(data_path, 'ephys.dat')

            num_channels = len(recording.continuous_files.keys()) if len(recording.continuous_files.keys()) > 0 else 43

            data = np.memmap(os.path.join(data_path, 'ephys.dat'), mode='r', dtype='int16')
            self.samples = data.reshape((len(data) // num_channels, num_channels))


    def __init__(self, data_path):

        self.mean_ripple_duration = 0

        file_list = os.listdir(data_path)

        self.continuous_files = {}

        for file_name in file_list:

            file_size = os.path.getsize(os.path.join(data_path, file_name))
            extension = os.path.splitext(file_name)[1]

            if extension == '.continuous':
                self.continuous_files[file_name] = file_size

        self.recording = analysis.Session(os.path.dirname(data_path)).recordnodes[0].recordings[0]

        self.sample_rate = 30000.0

        self.continuous = [ self.Continuous(self, data_path) ]
        
        self.load_events(data_path)

    def load_events(self, data_path):
            
        with open(os.path.join(data_path, 'messages.events'), 'r') as f:
            f.readline()
            self.first_sample_number = int(f.readline().split(' ')[0][:-1])
        self.first_timestamp = self.first_sample_number/self.sample_rate

        oe_events = self.recording.events
        events = oe_events.copy(deep=True)
        events['sample_number'] = events['sample_number'] - self.first_sample_number
        events['time (s)'] = events['sample_number']/self.sample_rate
        #print(events)
        #results.setdefault('First event',[]).append(events.sample_number[0])

        self.events = events

        #Load settings from settings.xml (TODO: add this helper to python-tools)
        import xml.etree.ElementTree as ET

        tree = ET.parse(os.path.join(data_path, 'settings.xml'))
        root = tree.getroot()

        channel_map_xml = root.findall('.//PROCESSOR[@name="Channel Map"]')[0].findall('.//CH')
        channel_order = []
        for elem in channel_map_xml:
            channel_order.extend([int(elem.attrib['index'])])
        head_size = 3
        tail_size = head_size
        self.channel_order = channel_order

        cnn_input_channels = []

        input = root.findall('.//PROCESSOR[@name="CNN-ripple"]')[0].findall('.//PARAMETERS')[0]
        cnn_input_channels.extend(ast.literal_eval(input.attrib['CNN_Input'])) 
        self.cnn_input_channels = cnn_input_channels

        self.num_detected_events = len(events)

        #Load manually identified ripple events
        text_file = open(os.path.join(data_path, os.path.join('events', 'events_selected_manually.txt')), "r")
        lines = text_file.readlines()[3:]
        text_file.close()
        events_selected_manually = []
        ripple_durations = []
        for line in lines:
            #The start of the event is defined near the first ripple or the sharp-wave onset
            start_timestamp = float(line.split()[0])
            #The end of the event is defined at the latest ripple or when sharp-wave resumed.
            stop_timestamp = float(line.split()[1])
            event_timestamp = int((start_timestamp + stop_timestamp)/2*self.sample_rate)
            ripple_durations.append((stop_timestamp - start_timestamp))
            #print(fmt_str.format('Ripple labeled:', event_timestamp, '['+str(start_timestamp)+','+str(stop_timestamp)+']'))
            events_selected_manually.append((start_timestamp, stop_timestamp))
        self.num_manually_labeled_events = len(events_selected_manually)
        self.labeled_bounds = events_selected_manually

        #Calculate average ripple duration
        mean_ripple_duration_ms = 1000*np.mean(ripple_durations)
        self.mean_ripple_duration_ms = mean_ripple_duration_ms