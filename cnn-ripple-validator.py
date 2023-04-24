from open_ephys import analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

import os 
import time
import ast

pd.set_option('display.max_rows', 40)

#Define base path to data
base_path = 'E:/PridaLabData/data_recieved_3_31_2023/2023-01-27_13-19-21_Threshold0_9/Record Node 115'

#List all file at base path
file_list = os.listdir(base_path)

continuous = {}

neural = {}
aux = {}
adc = {}

fmt_str = '{:<40} {:>18} {:>40}'
fmt_str_len = 40 + 18 + 30 + 2  # 2 spaces between each column ?

for file_name in file_list:
    file_size = os.path.getsize(os.path.join(base_path, file_name))
    extension = os.path.splitext(file_name)[1]
    #print(fmt_str.format(file_name, extension, file_size))
    if extension == '.continuous':
        continuous[file_name] = file_size
        if "AUX" in file_name:
            aux[file_name] = file_size
        elif "ADC" in file_name:
            adc[file_name] = file_size
        else:
            neural[file_name] = file_size

print('-' * fmt_str_len)

#Print total size of continuous files
print(fmt_str.format('Total size of continuous files ', '', sum(continuous.values())))

#Get size of .dat file
dat_file = os.path.join(base_path, 'ephys.dat')

#Print size of .dat file
print(fmt_str.format('Size of .dat file ', '', os.path.getsize(dat_file)))

#Missing bytes
print(fmt_str.format('Non data bytes ', '',  sum(continuous.values()) - os.path.getsize(dat_file)))

num_channels = len(continuous.keys())
#Load .dat file
data = np.memmap(os.path.join(base_path, 'ephys.dat'), mode='r', dtype='int16')
samples = data.reshape((len(data) // num_channels, num_channels))

samples_per_chan = int(samples.size/num_channels)
print(fmt_str.format('.dat shape:', '', '('+str(samples_per_chan)+','+str(num_channels)+')'))

#Print count of each file type
print(fmt_str.format('Total number of neural files', 'neural', len(neural.keys())))
print(fmt_str.format('Total number of aux files', 'aux', len(aux.keys())))
print(fmt_str.format('Total number of adc files', 'adc', len(adc.keys())))

print('-' * fmt_str_len)

#Load as open ephys object

#Start a timer
start_time = time.time()

session = analysis.Session(os.path.dirname(base_path))

record_node = session.recordnodes[0]

print(fmt_str.format('Total number of recordings', '', str(len(record_node.recordings))))

#print sample rate
print(fmt_str.format('Sample rate', '', str(record_node.recordings[0].continuous[0].metadata['sample_rate'])))

#print sample range
print(fmt_str.format('Sample range', '', str(record_node.recordings[0].continuous[0].sample_range)))

recording = record_node.recordings[0]

#Too slow takes 100 seconds to load the whole 0.82GB file
#print(fmt_str.format('Number of samples per file', '', str(recording.continuous[0].samples.shape[0])))

recording_time = float(samples_per_chan/recording.continuous[0].metadata['sample_rate'])

#Print total recording time in seconds
print(fmt_str.format('Total recording time (seconds)', '', str(round(recording_time,2))))

#Print total recording time in minutes
print(fmt_str.format('Total recording time (minutes)', '', str(round(recording_time/60,2))))


#Load the first 30 seconds of data
duration = 120
sample_rate = int(recording.continuous[0].metadata['sample_rate'])
end_sample_index = duration*sample_rate
# TODO: Way slower than loading .dat file directly, why?
#data = recording.continuous[0].get_samples(start_sample_index=0, end_sample_index=end_sample_index)

first_timestamp = recording.continuous[0].timestamps[0]
print(fmt_str.format('First timestamp', '', str(first_timestamp)))

first_sample_number = recording.continuous[0].sample_numbers[0]
print(fmt_str.format('First sample index', '', str(first_sample_number)))

print('-' * fmt_str_len)

#Load events
oe_events = recording.events
events = oe_events.copy()
events['sample_number'] = events['sample_number'] - first_sample_number
events['time (s)'] = events['sample_number']/sample_rate
print(fmt_str.format('Number of events', '', str(len(events))))
print(fmt_str.format('First event', '', str(events.sample_number[0])))
#print(events)

#Load settings.xml (TODO: add this helper to python-tools)
import xml.etree.ElementTree as ET

tree = ET.parse(os.path.join(base_path, 'settings.xml'))
root = tree.getroot()

#Get channel map        
channel_map_xml = root.findall('.//PROCESSOR[@name="Channel Map"]')[0].findall('.//CH')
channel_order = []
for elem in channel_map_xml:
    channel_order.extend([int(elem.attrib['index'])])

#Get cnn input channels
cnn_input_channels = []

input = root.findall('.//PROCESSOR[@name="CNN-ripple"]')[0].findall('.//PARAMETERS')[0]
cnn_input_channels.extend(ast.literal_eval(input.attrib['CNN_Input'])) 
print(fmt_str.format('CNN Input channels: ', '', str(cnn_input_channels)))

head_size = 3
tail_size = head_size
print(fmt_str.format('Channel ordering after channel map: ', '', str(channel_order[:head_size])[:-1]+' ... '+str(channel_order[-tail_size:])[1:]))

#Load manually identified ripple events
text_file = open(os.path.join(base_path, os.path.join('events', 'events_selected_manually.txt')), "r")
lines = text_file.readlines()[3:]
text_file.close()
events_selected_manually = []
ripple_durations = []
for line in lines:
    start_timestamp = float(line.split()[0])
    stop_timestamp = float(line.split()[1])
    event_timestamp = int((start_timestamp + stop_timestamp)/2*sample_rate)
    ripple_durations.append(stop_timestamp - start_timestamp)
    #print(fmt_str.format('Ripple labeled:', event_timestamp, '['+str(start_timestamp)+','+str(stop_timestamp)+']'))
    events_selected_manually.append(event_timestamp)
print(fmt_str.format('Number of manually labeled events: ', '', len(events_selected_manually)))
print(fmt_str.format('Average ripple duration (ms): ', '', round(1000*np.mean(ripple_durations),2)))

print(fmt_str.format('Elapsed time (seconds)', '', str(round(float(time.time() - start_time),2))))

PLOT_ALL = True
if PLOT_ALL:
    down_sample_factor = 100
    samples = samples + np.arange(samples.shape[1]) * 500
    plt.plot(samples[::down_sample_factor,cnn_input_channels])
    if events is not None:
        [plt.axvline(event_sample_number/down_sample_factor, 
                     color='b', linestyle='-', linewidth=1) for event_sample_number in events.sample_number]
    if events_selected_manually is not None:
        [plt.axvline(event_sample_number/down_sample_factor,
                     color='g', linestyle='-', linewidth=1) for event_sample_number in events_selected_manually]
    plt.axis('off')
    plt.show()

PLOT_FIRST_N_SNAPSHOTS = False
if PLOT_FIRST_N_SNAPSHOTS:
    N = 24
    window_size_in_ms = 200
    window_size_in_samples = int(window_size_in_ms * sample_rate / 1000)
    down_sample_factor = 1
    samples = samples + np.arange(samples.shape[1]) * 500
    #Create a sublplot for each event such that the resulting figure has roughly square plots
    num_rows = int(np.ceil(np.sqrt(N)))
    num_cols = int(np.ceil(N/num_rows))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*3,num_rows*3))
    print(axes.shape)
    for idx, ax in enumerate(axes.flatten()):
        #event_sample_number = events.sample_number[idx]
        event_sample_number = events_selected_manually[idx]
        ax.plot(samples[event_sample_number-window_size_in_samples:event_sample_number+window_size_in_samples:down_sample_factor,cnn_input_channels])
        ax.axvline(window_size_in_samples, color='b', linestyle='-', linewidth=1)
        ax.axis('off')
    plt.show()