from open_ephys.analysis import formats

from Plotter import SummaryPlot

import sys
import argparse

import pandas as pd
pd.set_option('display.max_rows', 40)
import numpy as np
import matplotlib.pyplot as plt 

from PridaRecording import PridaRecording

import os 
import time
import ast
import bisect

PLOT_SUMMARY                            = False
PLOT_FIRST_N_GROUND_TRUTH               = False
PLOT_FIRST_N_TRUE_POSITIVE              = True
PLOT_FIRST_N_FALSE_POSITIVE             = True
PLOT_TRUE_VS_FALSE_POSITIVE             = False

#Find all detected ripple events that are stricly contained within bounds of a manually labeled event
def find_contained_timestamps(ground_truth, detected):
    ground_truth = sorted(ground_truth, key=lambda x: x[0])

    def find_containing_bounds(detect_time):
        return next((bounds for bounds in ground_truth if bounds[0] <= detect_time <= bounds[1]), None)
    
    contained_timestamps = {detect_time: bounds for detect_time in detected if (bounds := find_containing_bounds(detect_time)) is not None}

    return contained_timestamps

#Find all detected ripple events that precede the start of a manually labeled bounds within n_milliseconds
def find_preceding_timestamps(ground_truth, detected, n_milliseconds):
    ground_truth = sorted(ground_truth, key=lambda x: x[0])

    def find_preceding_bounds(detect_time):
        return next((bounds for bounds in ground_truth if bounds[0] - n_milliseconds / 1000 <= detect_time < bounds[0]), None)
    
    preceding_timestamps = {detect_time: bounds for detect_time in detected if (bounds := find_preceding_bounds(detect_time)) is not None}

    return preceding_timestamps


#Find all false positive events 
def find_false_positive_timestamps(ground_truth, detected):
    ground_truth = sorted(ground_truth, key=lambda x: x[0])
    
    def is_false_positive(detect_time):
        return not any(bounds[0] <= detect_time <= bounds[1] for bounds in ground_truth)
    
    false_positive_timestamps = {detect_time: None for detect_time in detected if is_false_positive(detect_time)}
    
    return false_positive_timestamps

class Results(dict):
    def add(self, key, value):
        self.setdefault(key,[]).append(value)

def run(params):

    results = Results()

    prida_path = 'E:/PridaLabData/data_recieved_3_31_2023'
    local_path = 'D:/test-suite/ripples'

    data_sets = [
        os.path.join(prida_path, '2023-01-27_13-19-21_Threshold0_9/Record Node 115'), #prida recording 1
        #os.path.join(prida_path, '2023-01-27_13-29-15_Threshold0_9/Record Node 115'), #prida recording 2
        #os.path.join(prida_path, '2023-01-27_13-11-25_Threshold0_8/Record Node 115'), #prida recording 3

        os.path.join(local_path, '2023-04-30_23-48-28/Record Node 104/experiment1/recording1'), #replay recording 1
    ]

    count = 0

    recordings = []

    for data_path in data_sets:

        if 'PridaLab' in data_path:
            recording = PridaRecording(data_path)

            labeled_bounds = recording.labeled_bounds
            cnn_input_channels = [i - 1 for i in recording.cnn_input_channels]
            # Use the mean ripple duration as the threshold for predicting events
            pred_threshold = recording.mean_ripple_duration_ms
        else:
            recording = formats.BinaryRecording(data_path)

        # Get the sample rate 
        sample_rate = recording.continuous[0].metadata['sample_rate']
        
        #Only include events where state is 1
        recording.detected_events = recording.events[recording.events['state'] == 1]
        #print(recording.detected_events)

        recording.detected_events_within_manual_events = find_contained_timestamps(labeled_bounds, recording.detected_events.sample_number/sample_rate)
        recording.detected_events_not_within_manual_events = find_false_positive_timestamps(labeled_bounds, recording.detected_events.sample_number/sample_rate)
        recording.detected_events_within_pred_threshold = find_preceding_timestamps(labeled_bounds, recording.detected_events.sample_number/sample_rate, pred_threshold)

        results.add('Loaded dataset',       os.path.basename(data_path))
        results.add('Sample rate',          recording.continuous[0].metadata['sample_rate'])
        results.add('Number of channels',   np.shape(recording.continuous[0].samples)[1])
        results.add('Number of samples',    np.shape(recording.continuous[0].samples)[0])
        results.add('Duration (s)',         round(np.shape(recording.continuous[0].samples)[0] / recording.continuous[0].metadata['sample_rate'],2))
        results.add('CNN Input channels: ', str(cnn_input_channels))
        results.add('Number of events',     len(recording.detected_events))
        results.add('True positives',       len(recording.detected_events_within_manual_events))
        results.add('False positives',      len(recording.detected_events_not_within_manual_events))
        results.add('Predictive events',    len(recording.detected_events_within_pred_threshold))

        recordings.append(recording)

    PLOT = params['show']
    if PLOT:

        prida = recordings[0]
        local = recordings[1]

        raw = prida.continuous[0].samples
        filtered = local.continuous[0].samples

        all_events = prida.detected_events.sample_number

        if PLOT_SUMMARY:
            sp = SummaryPlot(
                data = filtered[:,cnn_input_channels],
                detected = all_events,
                ground_truth = [ int(np.mean([start,stop])*sample_rate) for start,stop in labeled_bounds],
                true_positives = [ key*sample_rate for key in prida.detected_events_within_manual_events.keys() ],
                false_positives = [ key*sample_rate for key in prida.detected_events_not_within_manual_events.keys() ],
                down_sample_factor = 10000
            )

        #TODO: Move the rest of these plots into Plotter class

        if PLOT_FIRST_N_GROUND_TRUTH:
            N = 16
            window_size_in_ms = 220
            window_size_in_samples = int(window_size_in_ms * sample_rate / 1000 / 2)
            down_sample_factor = 1
            samples = filtered + np.arange(filtered.shape[1]) * 400
            #Create a sublplot for each event such that the resulting figure has roughly square plots
            num_rows = int(np.ceil(np.sqrt(N)))
            num_cols = int(np.ceil(N/num_rows))
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*3,num_rows*3))
            gt_events = labeled_bounds
            for idx, ax in enumerate(axes.flatten()):
                offset = 0
                event_sample_number = int(sample_rate*np.mean(gt_events[idx+offset]))
                ax.plot(samples[event_sample_number-window_size_in_samples:event_sample_number+window_size_in_samples:down_sample_factor,cnn_input_channels], color='royalblue')
                x1 = int((gt_events[idx+offset][0] - event_sample_number/sample_rate) * sample_rate / down_sample_factor) + window_size_in_samples
                x2 = int((gt_events[idx+offset][1] - event_sample_number/sample_rate) * sample_rate / down_sample_factor) + window_size_in_samples
                ax.axvspan(x1, x2, facecolor='lightgray', alpha=0.5)
                ax.axis('off')
                ax.title.set_text(round(event_sample_number/sample_rate,2))
            plt.show()

        if PLOT_FIRST_N_TRUE_POSITIVE:
            N = 16
            down_sample_factor = 1
            window_size_in_ms = 220 / down_sample_factor
            window_size_in_samples = int(window_size_in_ms * sample_rate / 1000 / 2)
            samples = filtered + np.arange(filtered.shape[1]) * 400
            #Create a sublplot for each event such that the resulting figure has roughly square plots
            num_rows = int(np.ceil(np.sqrt(N)))
            num_cols = int(np.ceil(N/num_rows))
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*3,num_rows*3))
            true_positives = list(prida.detected_events_within_manual_events.keys())
            for idx, ax in enumerate(axes.flatten()):
                offset = 0
                event_sample_number = int(sample_rate*true_positives[offset+idx])
                ax.plot(samples[event_sample_number-window_size_in_samples:event_sample_number+window_size_in_samples:down_sample_factor,cnn_input_channels], color='green')
                ax.axvline(window_size_in_samples, color='k', linestyle='-', linewidth=2)
                bounds = prida.detected_events_within_manual_events[true_positives[offset+idx]]
                x1 = int((bounds[0] - event_sample_number/sample_rate) * sample_rate / down_sample_factor) + window_size_in_samples
                x2 = int((bounds[1] - event_sample_number/sample_rate) * sample_rate / down_sample_factor) + window_size_in_samples
                ax.axvspan(x1, x2, facecolor='lightgray', alpha=0.5)
                ax.axis('off')
                ax.title.set_text(round(event_sample_number/sample_rate,2))
            plt.show()

        if PLOT_FIRST_N_FALSE_POSITIVE:
            N = 16
            down_sample_factor = 1
            window_size_in_ms = 220 / down_sample_factor
            window_size_in_samples = int(window_size_in_ms * sample_rate / 1000 / 2)
            samples = filtered + np.arange(filtered.shape[1]) * 400
            #Create a sublplot for each event such that the resulting figure has roughly square plots
            num_rows = int(np.ceil(np.sqrt(N)))
            num_cols = int(np.ceil(N/num_rows))
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*3,num_rows*3))
            false_positives = list(prida.detected_events_not_within_manual_events.keys())
            for idx, ax in enumerate(axes.flatten()):
                offset = 0
                event_sample_number = int(sample_rate*false_positives[offset+idx])
                ax.plot(samples[event_sample_number-window_size_in_samples:event_sample_number+window_size_in_samples:down_sample_factor,cnn_input_channels], color='red')
                ax.axvline(window_size_in_samples, color='k', linestyle='-', linewidth=2)
                ax.axis('off')
                ax.title.set_text(round(event_sample_number/sample_rate,2))
            plt.show()

    return results

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Validate CNN-based ripple detection')
    parser.add_argument('--show', required=False, type=int, default=0)
    parser.add_argument('--count', required=False, type=int, default=3)
    params = vars(parser.parse_args(sys.argv[1:]))

    results = run(params)

    fmt = '{:<40} ' + '{:>40}' * params['count']
    print(fmt.format('DESCRIPTION', 'PRIDA LAB', 'OPEN EPHYS'))
    print('-'*(40*(params['count'] + 1) + params['count']))
    for key, val in results.items():
        print(fmt.format(key, *val))