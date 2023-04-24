from open_ephys import analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

import os 
import time
import ast
import bisect

pd.set_option('display.max_rows', 40)

#This script aims to validate claims in paper: https://elifesciences.org/articles/77772
# 80% of predictions were consistent with the ground truth


duration_threshold = 100

#Define base path to data
base_path = 'E:/PridaLabData/data_recieved_3_31_2023'
data_sets = [
    os.path.join(base_path, '2023-01-27_13-19-21_Threshold0_9/Record Node 115'),
    os.path.join(base_path, '2023-01-27_13-29-15_Threshold0_9/Record Node 115'),
    os.path.join(base_path, '2023-01-27_13-11-25_Threshold0_8/Record Node 115')
]

data_path = data_sets[0]

#List all file at base path
file_list = os.listdir(data_path)

continuous = {}

neural = {}
aux = {}
adc = {}

fmt_str = '{:<40} {:>18} {:>40}'
fmt_str_len = 40 + 18 + 40 + 2  # 2 spaces between each column ?

for file_name in file_list:
    file_size = os.path.getsize(os.path.join(data_path, file_name))
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
#print(fmt_str.format('Total size of continuous files ', '', sum(continuous.values())))

#Get size of .dat file
dat_file = os.path.join(data_path, 'ephys.dat')

#Print size of .dat file
#print(fmt_str.format('Size of .dat file ', '', os.path.getsize(dat_file)))

#Missing bytes
#print(fmt_str.format('Non data bytes ', '',  sum(continuous.values()) - os.path.getsize(dat_file)))

num_channels = len(continuous.keys()) if len(continuous.keys()) > 0 else 43

#Load .dat file
data = np.memmap(os.path.join(data_path, 'ephys.dat'), mode='r', dtype='int16')
samples = data.reshape((len(data) // num_channels, num_channels))

samples_per_chan = int(samples.size/num_channels)
#print(fmt_str.format('.dat shape:', '', '('+str(samples_per_chan)+','+str(num_channels)+')'))

#Print count of each file type
#print(fmt_str.format('Total number of neural files', 'neural', len(neural.keys())))
#print(fmt_str.format('Total number of aux files', 'aux', len(aux.keys())))
#print(fmt_str.format('Total number of adc files', 'adc', len(adc.keys())))

#print('-' * fmt_str_len)

#Load as open ephys object

#Start a timer
start_time = time.time()

session = analysis.Session(os.path.dirname(data_path))

record_node = session.recordnodes[0]

print(fmt_str.format('Total number of recordings', '', str(len(record_node.recordings))))

#print sample rate
sample_rate = record_node.recordings[0].continuous[0].metadata['sample_rate'] if len(continuous.keys()) > 0 else 30000

print(fmt_str.format('Sample rate', '', str(sample_rate)))

recording = record_node.recordings[0]

#Too slow takes 100 seconds to load the whole 0.82GB file
#print(fmt_str.format('Number of samples per file', '', str(recording.continuous[0].samples.shape[0])))

recording_time = float(samples_per_chan/sample_rate)

#Print total recording time in seconds
print(fmt_str.format('Total recording time (seconds)', '', str(round(recording_time,2))))

#Print total recording time in minutes
print(fmt_str.format('Total recording time (minutes)', '', str(round(recording_time/60,2))))


#Load the first 30 seconds of data
duration = 120
end_sample_index = duration*sample_rate
# TODO: Way slower than loading .dat file directly, why?
#data = recording.continuous[0].get_samples(start_sample_index=0, end_sample_index=end_sample_index)

if len(continuous.keys()) > 0:
    first_sample_number = recording.continuous[0].sample_numbers[0]
    first_timestamp = recording.continuous[0].timestamps[0]
else:
    #Get first timestamp from the first word in second line of messages.events file
    with open(os.path.join(base_path, 'messages.events'), 'r') as f:
        f.readline()
        first_sample_number = int(f.readline().split(' ')[0][:-1])
    first_timestamp = first_sample_number/sample_rate

print(fmt_str.format('First timestamp', '', str(first_timestamp)))
print(fmt_str.format('First sample number', '', str(first_sample_number)))

print('-' * fmt_str_len)

#Load events
oe_events = recording.events
events = oe_events.copy(deep=True)
events['sample_number'] = events['sample_number'] - first_sample_number
events['time (s)'] = events['sample_number']/sample_rate
print(fmt_str.format('Number of events', '', str(len(events))))
print(fmt_str.format('First event', '', str(events.sample_number[0])))
#print(events)

#Load settings.xml (TODO: add this helper to python-tools)
import xml.etree.ElementTree as ET

tree = ET.parse(os.path.join(data_path, 'settings.xml'))
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

print(fmt_str.format('Number of detected events: ', '', str(len(events))))

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
    event_timestamp = int((start_timestamp + stop_timestamp)/2*sample_rate)
    ripple_durations.append((stop_timestamp - start_timestamp))
    #print(fmt_str.format('Ripple labeled:', event_timestamp, '['+str(start_timestamp)+','+str(stop_timestamp)+']'))
    events_selected_manually.append((start_timestamp, stop_timestamp))
print(fmt_str.format('Number of manually labeled events: ', '', len(events_selected_manually)))

#Calculate average ripple duration
mean_ripple_duration_ms = 1000*np.mean(ripple_durations)
print(fmt_str.format('Average ripple duration (ms): ', '', round(mean_ripple_duration_ms,2)))

#Compare events to manually labeled events

#Find events that are stricly contained within a manually labeled event
def find_contained_timestamps(ground_truth, predicted):
    ground_truth = sorted(ground_truth, key=lambda x: x[0], reverse=False)
    contained_timestamps = {}

    for pred_time in predicted:
        for bounds in ground_truth:
            if pred_time >= bounds[0] and pred_time <= bounds[1]:
                contained_timestamps[pred_time] = bounds
                break

    return contained_timestamps

#Find events that precede the start of a manually labeled event within n_milliseconds
def find_preceding_timestamps(ground_truth, predicted, n_milliseconds):
    ground_truth = sorted(ground_truth, key=lambda x: x[0], reverse=False)
    preceding_timestamps = []

    for pred_time in predicted:
        for bounds in ground_truth:
            if pred_time >= bounds[0] - n_milliseconds / 1000 and pred_time < bounds[0]:
                preceding_timestamps.append(pred_time)
                break

    return preceding_timestamps

detected_events_within_manual_events = find_contained_timestamps(events_selected_manually, events.sample_number/sample_rate)
print(fmt_str.format('Events inside labeled bounds: ', '', str(len(detected_events_within_manual_events.keys()))))
pred_threshold = 200
detected_events_within_pred_threshold = find_preceding_timestamps(events_selected_manually, events.sample_number/sample_rate, pred_threshold)
print(fmt_str.format('Events '+str(pred_threshold)+' ms before labeled bounds: ', '', str(len(detected_events_within_pred_threshold))))

print(fmt_str.format('True positive rate: ', '', str(round(len(detected_events_within_manual_events.keys())/len(events_selected_manually),2))))
print(fmt_str.format('False positive rate: ', '', str(round(1-len(detected_events_within_manual_events.keys())/len(events_selected_manually),2))))
print(fmt_str.format('Elapsed time (seconds)', '', str(round(float(time.time() - start_time),2))))

PLOT = False
if PLOT:
    PLOT_SUMMARY = True
    if PLOT_SUMMARY:
        down_sample_factor = 10000
        samples = samples + np.arange(samples.shape[1]) * 500
        plt.plot(samples[::down_sample_factor,cnn_input_channels],color='0.5')
        if events is not None:
            [plt.axvline(event_sample_number/down_sample_factor, 
                        color='b', linestyle='-', linewidth=1) for event_sample_number in events.sample_number]
        if events_selected_manually is not None:
            [plt.axvline(int(event_sample_number/down_sample_factor),
                        color='r', linestyle='-', linewidth=1) for event_sample_number in [ int(np.mean([a,b])*sample_rate) for a,b in events_selected_manually]]
        if detected_events_within_manual_events is not None:
            [plt.axvline(int(sample_rate*event_sample_number)/down_sample_factor,
                        color='g', linestyle='-', linewidth=1) for event_sample_number in detected_events_within_manual_events]
        if detected_events_within_pred_threshold is not None:
            [plt.axvline(int(sample_rate*event_sample_number)/down_sample_factor,
                        color='g', linestyle='-', linewidth=1) for event_sample_number in detected_events_within_pred_threshold]
        plt.axis('off')
        plt.show()

    PLOT_FIRST_N_GROUND_TRUTH = False
    if PLOT_FIRST_N_GROUND_TRUTH:
        N = 16
        window_size_in_ms = 150
        window_size_in_samples = int(window_size_in_ms * sample_rate / 1000 / 2)
        down_sample_factor = 1
        samples = samples + np.arange(samples.shape[1]) * 150
        #Create a sublplot for each event such that the resulting figure has roughly square plots
        num_rows = int(np.ceil(np.sqrt(N)))
        num_cols = int(np.ceil(N/num_rows))
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*3,num_rows*3))
        gt_events = events_selected_manually
        for idx, ax in enumerate(axes.flatten()):
            event_sample_number = int(sample_rate*np.mean(gt_events[idx]))
            ax.plot(samples[event_sample_number-window_size_in_samples:event_sample_number+window_size_in_samples:down_sample_factor,cnn_input_channels], color='royalblue')
            x1 = int((gt_events[idx][0] - event_sample_number/sample_rate) * sample_rate / down_sample_factor) + window_size_in_samples
            x2 = int((gt_events[idx][1] - event_sample_number/sample_rate) * sample_rate / down_sample_factor) + window_size_in_samples
            ax.axvspan(x1, x2, facecolor='lightgray', alpha=0.5)
            ax.axis('off')
        plt.show()

    PLOT_FIRST_N_GROUND_TRUTH_AND_DETECTED = True
    if PLOT_FIRST_N_GROUND_TRUTH_AND_DETECTED:
        N = 32
        window_size_in_ms = 120
        window_size_in_samples = int(window_size_in_ms * sample_rate / 1000 / 2)
        down_sample_factor = 1
        samples = samples + np.arange(samples.shape[1]) * 300
        #Create a sublplot for each event such that the resulting figure has roughly square plots
        num_rows = int(np.ceil(np.sqrt(N)))
        num_cols = int(np.ceil(N/num_rows))
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*3,num_rows*3))
        gt_events = list(detected_events_within_manual_events.keys())
        for idx, ax in enumerate(axes.flatten()):
            event_sample_number = int(sample_rate*gt_events[idx])
            ax.plot(samples[event_sample_number-window_size_in_samples:event_sample_number+window_size_in_samples:down_sample_factor,cnn_input_channels], color='royalblue')
            ax.axvline(window_size_in_samples, color='k', linestyle='-', linewidth=2)
            x1 = int((detected_events_within_manual_events[gt_events[idx]][0] - event_sample_number/sample_rate) * sample_rate / down_sample_factor) + window_size_in_samples
            x2 = int((detected_events_within_manual_events[gt_events[idx]][1] - event_sample_number/sample_rate) * sample_rate / down_sample_factor) + window_size_in_samples
            ax.axvspan(x1, x2, facecolor='lightgray', alpha=0.5)
            ax.axis('off')
        plt.show()