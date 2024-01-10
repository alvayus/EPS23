import wfdb
import matplotlib.pyplot as plt
import numpy as np

n_secs = 3
freq = 360

#record = wfdb.rdrecord('data/212', sampto=n_secs*freq)
#ann = wfdb.rdann('data/212', 'atr', sampto=n_secs*freq)
#wfdb.plot_wfdb(record, annotation = ann)

# Read sample
signals, fields = wfdb.rdsamp('data/117', sampto=int(n_secs*freq))

# Delta modulator
mlii = signals[:, 0]

dc = 0
delta = 0.03
on_spikes = []
off_spikes = []

for i in range(len(mlii)):
    current_sample = mlii[i]

    if current_sample > dc + delta:
        dc = current_sample

        time = i / freq  # Extract current time
        on_spikes.append(time)

    if current_sample < dc - delta:
        dc = current_sample

        time = i / freq  # Extract current time
        off_spikes.append(time)

plot_mlii = []
for i in range(len(mlii)):
    plot_mlii.append([i / freq, mlii[i]])
plot_mlii = np.array(plot_mlii)
min_mlii = min(mlii)

# Plots
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = '12'
plt.rcParams["figure.figsize"] = (12, 3.5)

fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
fig.suptitle('ECG representation')

axs[0].plot(plot_mlii[:, 0], plot_mlii[:, 1])
axs[0].set_xlim([0, n_secs])
#axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('MLII (mV)')

axs[1].plot(on_spikes, [2] * len(on_spikes), marker='|', linestyle='None', color='teal')
axs[1].axhline(y=2, linewidth=1, color='teal')
axs[1].plot(off_spikes, [1] * len(off_spikes), marker='|', linestyle='None', color='palevioletred')
axs[1].axhline(y=1, linewidth=1, color='palevioletred')
axs[1].set_xlim([0, n_secs])
axs[1].set_ylim([0, 3])
axs[1].set_yticks([1, 2])
axs[1].set_yticklabels(["OFF", "ON"])
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Delta mod.')

plt.tight_layout()
plt.show()