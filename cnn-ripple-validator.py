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

PLOT_SUMMARY                            = False
PLOT_FIRST_N_GROUND_TRUTH               = False
PLOT_FIRST_N_TRUE_POSITIVE              = False
PLOT_FIRST_N_FALSE_POSITIVE             = False
PLOT_TRUE_VS_FALSE_POSITIVE             = True

def run(params):

    results = {};

    duration_threshold = 100

    base_path = 'E:/PridaLabData/data_recieved_3_31_2023'
    data_sets = [
        os.path.join(base_path, '2023-01-27_13-19-21_Threshold0_9/Record Node 115'),
        os.path.join(base_path, '2023-01-27_13-29-15_Threshold0_9/Record Node 115'),
        os.path.join(base_path, '2023-01-27_13-11-25_Threshold0_8/Record Node 115')
    ]

    count = 0

    for data_path in data_sets:

        if count > params['count']:
            break

        count += 1

        file_list = os.listdir(data_path)

        continuous = {}

        neural = {}
        aux = {}
        adc = {}

        for file_name in file_list:

            file_size = os.path.getsize(os.path.join(data_path, file_name))
            extension = os.path.splitext(file_name)[1]

            if extension == '.continuous':
                continuous[file_name] = file_size
                if "AUX" in file_name:
                    aux[file_name] = file_size
                elif "ADC" in file_name:
                    adc[file_name] = file_size
                else:
                    neural[file_name] = file_size

        dat_file = os.path.join(data_path, 'ephys.dat')

        num_channels = len(continuous.keys()) if len(continuous.keys()) > 0 else 43

        data = np.memmap(os.path.join(data_path, 'ephys.dat'), mode='r', dtype='int16')
        samples = data.reshape((len(data) // num_channels, num_channels))

        samples_per_chan = int(samples.size/num_channels)

        start_time = time.time()

        session = analysis.Session(os.path.dirname(data_path))

        record_node = session.recordnodes[0]

        results.setdefault('Number of recordings',[]).append(len(record_node.recordings))

        sample_rate = record_node.recordings[0].continuous[0].metadata['sample_rate'] if len(continuous.keys()) > 0 else 30000

        results.setdefault('Sample rate',[]).append(sample_rate)

        recording = record_node.recordings[0]

        #Too slow takes 100 seconds to load the whole 0.82GB file
        #print(fmt_str.format('Number of samples per file', '', str(recording.continuous[0].samples.shape[0])))

        recording_time = float(samples_per_chan/sample_rate)
        results.setdefault('Total recording time (minutes)',[]).append(round(recording_time/60,2))

        duration = 120 #seconds
        end_sample_index = duration*sample_rate

        # TODO: Way slower than loading .dat file directly, why?
        #data = recording.continuous[0].get_samples(start_sample_index=0, end_sample_index=end_sample_index)

        if len(continuous.keys()) > 0:
            first_sample_number = recording.continuous[0].sample_numbers[0]
            first_timestamp = recording.continuous[0].timestamps[0]
        else:
            #Get first timestamp from the first word in second line of messages.events file
            with open(os.path.join(data_path, 'messages.events'), 'r') as f:
                f.readline()
                first_sample_number = int(f.readline().split(' ')[0][:-1])
            first_timestamp = first_sample_number/sample_rate

        #results.setdefault('First timestamp',[]).append(first_timestamp)
        #results.setdefault('First sample number',[]).append(first_sample_number)

        oe_events = recording.events
        events = oe_events.copy(deep=True)
        events['sample_number'] = events['sample_number'] - first_sample_number
        events['time (s)'] = events['sample_number']/sample_rate
        #results.setdefault('First event',[]).append(events.sample_number[0])

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
        results.setdefault('Channel ordering after channel map',[]).append(str(channel_order[:head_size])[:-1]+' ... '+str(channel_order[-tail_size:])[1:])

        cnn_input_channels = []

        input = root.findall('.//PROCESSOR[@name="CNN-ripple"]')[0].findall('.//PARAMETERS')[0]
        cnn_input_channels.extend(ast.literal_eval(input.attrib['CNN_Input'])) 
        results.setdefault('CNN Input channels',[]).append(str(cnn_input_channels))

        results.setdefault('Number of detected events',[]).append(len(events))

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
        results.setdefault('Number of manually labeled events',[]).append(len(events_selected_manually))

        #Calculate average ripple duration
        mean_ripple_duration_ms = 1000*np.mean(ripple_durations)
        results.setdefault('Average ripple duration (ms)',[]).append(round(mean_ripple_duration_ms,2))

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
            preceding_timestamps = {}

            for pred_time in predicted:
                for bounds in ground_truth:
                    if pred_time >= bounds[0] - n_milliseconds / 1000 and pred_time < bounds[0]:
                        preceding_timestamps[pred_time] = bounds
                        break

            return preceding_timestamps

        #Find false positive events
        def find_false_positive_timestamps(ground_truth, predicted):
            ground_truth.sort(key=lambda x: x[0])
            ground_truth_start_times = [bounds[0] for bounds in ground_truth]
            false_positive_timestamps = {}

            for pred_time in predicted:
                index = bisect.bisect_left(ground_truth_start_times, pred_time)
                # If the index is 0, the predicted time is not within any interval
                if index == 0:
                    false_positive_timestamps[pred_time] = 1
                else:
                    previous_interval = ground_truth[index - 1]
                    if pred_time > previous_interval[1]:  # pred_time is not in the previous interval
                        false_positive_timestamps[pred_time] = 1

            return false_positive_timestamps

        detected_events_within_manual_events = find_contained_timestamps(events_selected_manually, events.sample_number/sample_rate)
        results.setdefault('Events inside labeled bounds',[]).append(len(detected_events_within_manual_events.keys()))

        detected_events_not_within_manual_events = find_false_positive_timestamps(events_selected_manually, events.sample_number/sample_rate)
        results.setdefault('Events outside labeled bounds',[]).append(len(detected_events_not_within_manual_events.keys()))

        pred_threshold = mean_ripple_duration_ms
        detected_events_within_pred_threshold = find_preceding_timestamps(events_selected_manually, events.sample_number/sample_rate, pred_threshold)
        results.setdefault('Predict windows (in ms)',[]).append(round(pred_threshold,1))
        results.setdefault('Events detected within predict window:',[]).append(len(detected_events_within_pred_threshold.keys()))
        
        #Types of events:
        #sharp waves without clear associated ripples (SW no ripples)
        #ripples without associated sharp waves (ripples no SW),

        #True positives: events that are stricly contained within a manually labeled event
        #False positives: events that are not stricly contained within a manually labeled event nor precede the start of a manually labeled event within n_milliseconds

        results.setdefault('True positives',[]).append(len(detected_events_within_manual_events.keys()))
        results.setdefault('False positives',[]).append(len(detected_events_not_within_manual_events.keys()))
        results.setdefault('Elapsed time (seconds)',[]).append(round(float(time.time() - start_time),2))

        PLOT = params['show']
        if PLOT:
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

            if PLOT_FIRST_N_TRUE_POSITIVE:
                N = 32
                down_sample_factor = 1
                window_size_in_ms = 120 / down_sample_factor
                window_size_in_samples = int(window_size_in_ms * sample_rate / 1000 / 2)
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
                    ax.set_title(str(idx))
                plt.show()

            if PLOT_FIRST_N_FALSE_POSITIVE:
                N = 32
                down_sample_factor = 1
                window_size_in_ms = 120 / down_sample_factor
                window_size_in_samples = int(window_size_in_ms * sample_rate / 1000 / 2)
                samples = samples + np.arange(samples.shape[1]) * 300
                #Create a sublplot for each event such that the resulting figure has roughly square plots
                num_rows = int(np.ceil(np.sqrt(N)))
                num_cols = int(np.ceil(N/num_rows))
                fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*3,num_rows*3))
                false_positives = list(detected_events_not_within_manual_events.keys())
                for idx, ax in enumerate(axes.flatten()):
                    event_sample_number = int(sample_rate*false_positives[idx])
                    ax.plot(samples[event_sample_number-window_size_in_samples:event_sample_number+window_size_in_samples:down_sample_factor,cnn_input_channels], color='royalblue')
                    ax.axvline(window_size_in_samples, color='k', linestyle='-', linewidth=2)
                    ax.axis('off')
                    ax.set_title(str(idx))
                plt.show()

            if PLOT_TRUE_VS_FALSE_POSITIVE:
                N = 16
                down_sample_factor = 1
                window_size_in_ms = 120 / down_sample_factor
                window_size_in_samples = int(window_size_in_ms * sample_rate / 1000 / 2)
                samples = samples + np.arange(samples.shape[1]) * 300
                #Create a sublplot for each event such that the resulting figure has roughly square plots
                num_rows = int(np.ceil(np.sqrt(N)))
                num_cols = int(np.ceil(N/num_rows))
                fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*3,num_rows*3))
                gt_events = list(detected_events_within_manual_events.keys())
                true_positives = list(detected_events_within_manual_events.keys())
                false_positives = list(detected_events_not_within_manual_events.keys())
                for idx, ax in enumerate(axes.flatten()):
                    if idx < N / 2:
                        event_sample_number = int(sample_rate*true_positives[idx])
                        ax.plot(samples[event_sample_number-window_size_in_samples:event_sample_number+window_size_in_samples:down_sample_factor,cnn_input_channels], color='royalblue')
                    else:
                        event_sample_number = int(sample_rate*false_positives[int(idx - N / 2)])
                        ax.plot(samples[event_sample_number-window_size_in_samples:event_sample_number+window_size_in_samples:down_sample_factor,cnn_input_channels], color='red')
                    ax.axvline(window_size_in_samples, color='grey', linestyle='-', linewidth=1)
                    ax.axis('off')
                plt.show()

    return results

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Validate CNN-based ripple detection')
    parser.add_argument('--show', required=False, type=int, default=0)
    parser.add_argument('--count', required=False, type=int, default=3)

    params = vars(parser.parse_args(sys.argv[1:]))

    results = run(params)

    fmt = '{:<40} ' + '{:>40}' * params['count']
    print(fmt.format('DESCRIPTION', 'DATASET 1', 'DATASET 2', 'DATASET 3'))
    print('-'*(40*(params['count'] + 1) + params['count']))
    for key, val in results.items():
        print(fmt.format(key, *val))