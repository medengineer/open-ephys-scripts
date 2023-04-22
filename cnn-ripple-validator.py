from open_ephys import analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

import os 
import time

pd.set_option('display.max_rows', 40)

#Define base path to data
base_path = 'E:/PridaLabData/data_recieved_3_31_2023/2023-01-27_13-19-21_Threshold0_9/Record Node 115'

#List all file at base path
file_list = os.listdir(base_path)

continuous = {}

neural = {}
aux = {}
adc = {}

fmt_str = '{:<40} {:>18} {:>30}'
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
events = recording.events
print(fmt_str.format('Number of events', '', str(len(events))))
print(fmt_str.format('First event', '', str(events.sample_number[0])))
print(events)

print(fmt_str.format('Elapsed time (seconds)', '', str(round(float(time.time() - start_time),2))))

#Display results
PLOT = True
if PLOT:
    down_sample_factor = 10000
    samples = samples + np.arange(samples.shape[1]) * 500
    plt.plot(samples[::down_sample_factor,:32])
    if events is not None:
        [plt.axvline((event_sample_number - first_sample_number)/down_sample_factor, 
                     color='b', linestyle='-', linewidth=1) for event_sample_number in events.sample_number]
    plt.axis('off')
    plt.show()