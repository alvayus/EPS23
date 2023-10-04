import wfdb
import matplotlib.pyplot as plt
import numpy as np

n_secs = 5
freq = 360

record = wfdb.rdrecord('data/100', sampto=n_secs*freq)
ann = wfdb.rdann('data/100', 'atr', sampto=n_secs*freq)
wfdb.plot_wfdb(record, annotation = ann)

# Read sample
signals, fields = wfdb.rdsamp('data/100', sampto=n_secs*freq)

# Delta modulator
v5 = signals[:,1]

dc = 0
delta = 0.01
on_spikes = []
off_spikes = []

for i in range(len(v5)):
    current_sample = v5[i]

    if current_sample > dc + delta:
        dc = current_sample

        time = i / freq  # Extract current time
        on_spikes.append(time)

    if current_sample < dc - delta:
        dc = current_sample

        time = i / freq  # Extract current time
        off_spikes.append(time)

plot_v5 = []
for i in range(len(v5)):
    plot_v5.append([i / freq, v5[i]])
plot_v5 = np.array(plot_v5)
min_mlii = min(v5)

# Plots
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = '12'
plt.rcParams["figure.figsize"] = (12, 6)

fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
fig.suptitle('ECG representation')

axs[0].plot(plot_v5[:, 0], plot_v5[:, 1])
axs[0].set_xlim([0, n_secs])
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('V5 (mV)')

axs[1].plot(on_spikes, [2] * len(on_spikes), marker='|', linestyle=None, color='teal')
axs[1].plot(off_spikes, [1] * len(off_spikes), marker='|', linestyle=None, color='palevioletred')
axs[1].set_xlim([0, n_secs])
axs[1].set_ylim([0, 3])
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Delta mod.')

plt.tight_layout()
plt.show()