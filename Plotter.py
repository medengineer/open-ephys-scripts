import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import math

from matplotlib import gridspec
class RipplePlot:

    def __init__(self, axes_handle):
        self.ax = axes_handle
        self.num_datasets = 8
        self.num_points = 100
        self.amplitude = 5

        datasets = [self.amplitude * np.sin(np.linspace(0, 2 * np.pi, self.num_points)) + i * 2 * self.amplitude for i in range(self.num_datasets)]

        # Plot each dataset along the Z-axis
        for i, data in enumerate(datasets):
            x = np.arange(self.num_points)
            y = np.ones(self.num_points) * i
            z = data
            self.ax.plot(x, y, z)
            self.ax.axis('off')

N = 8
cols = 4
rows = int(math.ceil(N / cols))

gs = gridspec.GridSpec(rows, cols)
fig = plt.figure()
for n in range(N):
    ax = fig.add_subplot(gs[n], projection='3d')
    RipplePlot(ax)

fig.tight_layout()
plt.show()

#TODO Refactor the below from main script
'''

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
                window_size_in_ms = 220
                window_size_in_samples = int(window_size_in_ms * sample_rate / 1000 / 2)
                down_sample_factor = 1
                samples = samples + np.arange(samples.shape[1]) #* 400
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
                    ax.title.set_text(round(event_sample_number/sample_rate,2))
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
                    ax.title.set_text(round(event_sample_number/sample_rate,2))
                plt.show()

            if PLOT_FIRST_N_FALSE_POSITIVE:
                N = 8
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
                    ax.title.set_text(round(event_sample_number/sample_rate,2))
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
                print(true_positives)
                false_positives = list(detected_events_not_within_manual_events.keys())
                print(false_positives)
                for idx, ax in enumerate(axes.flatten()):
                    if idx < N / 2:
                        event_sample_number = int(sample_rate*true_positives[idx])
                        ax.plot(samples[event_sample_number-window_size_in_samples:event_sample_number+window_size_in_samples:down_sample_factor,cnn_input_channels], color='royalblue')
                    else:
                        event_sample_number = int(sample_rate*false_positives[int(idx - N / 2)])
                        ax.plot(samples[event_sample_number-window_size_in_samples:event_sample_number+window_size_in_samples:down_sample_factor,cnn_input_channels], color='red')
                    ax.axvline(window_size_in_samples, color='grey', linestyle='-', linewidth=1)
                    ax.title.set_text(round(event_sample_number/sample_rate,2))
                    ax.axis('off')
                plt.show()
'''