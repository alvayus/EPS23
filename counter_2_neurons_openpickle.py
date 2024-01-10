import pickle
import matplotlib.pyplot as plt

global_params = {"min_delay": 1.0, "sim_time": 150.0}
n_bits = 4

filename = "counter_2_neurons_1"
with open("experiments/" + filename + '.pickle', 'rb') as f:
    load_array = pickle.load(f)

switch_data = load_array[0]
and_data = load_array[1]
count_times = load_array[2]

# --- Saving plot ---
plt.rcParams['figure.dpi'] = 400
plt.rcParams['font.size'] = '4'
plt.rcParams["figure.figsize"] = (4, 0.9)

fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [9, 1.75]}, sharex=True)
#fig.suptitle('Spiking response')

# Spikes
n_id = 0
for segment in switch_data:
    n_tmp = len(segment.spiketrains)
    for i in range(n_tmp):
        axs[0].plot(segment.spiketrains[i], [n_id] * len(segment.spiketrains[i]), 'o', markersize=0.5, color='darkmagenta')
        n_id += 1

'''for segment in and_data:
    n_tmp = len(segment.spiketrains)
    for i in range(n_tmp):
        axs[0].plot(segment.spiketrains[i], [n_id] * len(segment.spiketrains[i]), 'o', markersize=0.5, color='darkmagenta')
        n_id += 1'''

axs[0].set_xlim([0, global_params["sim_time"]])
'''axs[0].set_ylim([-1, 3 * n_bits + 3 * (n_bits - 1)])
axs[0].set_yticks(range(3 * n_bits + 3 * (n_bits - 1)))'''
axs[0].set_ylim([-1, 3 * n_bits])
axs[0].set_yticks(range(0, 3 * n_bits, 3))
#axs[0].set_xlabel('Time (ms)')
axs[0].set_ylabel('Neuron IDs')

# Inputs
axs[1].plot(count_times, [0] * len(count_times), 'o', markersize=0.5, color='orange')
axs[1].set_xlim([0, global_params["sim_time"]])
axs[1].set_ylim([-1, 1])
axs[1].set_yticks([0], labels=[" "])
axs[1].set_xlabel('Time (ms)')
axs[1].set_ylabel('Input')

plt.tight_layout()
plt.savefig("experiments/" + filename + '.png', transparent=False, facecolor='white', edgecolor='black')
plt.show()