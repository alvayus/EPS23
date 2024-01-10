import pickle
import matplotlib.pyplot as plt

global_params = {"min_delay": 1.0, "sim_time": 150.0}
n_bits = 2

filename = "counter_2_neurons_ecg_1"
with open("experiments/" + filename + '.pickle', 'rb') as f:
    load_array = pickle.load(f)

switch_data = load_array[0]
and_data = load_array[1]
all_spikes = load_array[2]
oc_data = load_array[3]

# --- Saving plot ---
plt.rcParams['figure.dpi'] = 400
plt.rcParams['font.size'] = '4'
plt.rcParams["figure.figsize"] = (4, 1.2)

fig, axs = plt.subplots(3, 1, gridspec_kw={'height_ratios': [2, 6, 1]}, sharex=True)
# fig.suptitle('Spiking response')

# Overflow counter
axs[0].plot(oc_data[0].spiketrains[0], [0] * len(oc_data[0].spiketrains[0]), 'o', markersize=0.5, color='darkmagenta')
axs[0].plot(oc_data[1].spiketrains[0], [1] * len(oc_data[1].spiketrains[0]), 'o', markersize=0.5, color='palevioletred')
axs[0].set_xlim([0, global_params["sim_time"]])
axs[0].set_ylim([-1, 2])
axs[0].set_yticks([0, 1], labels=["CO", "FO"])  # Counter output, Filter output
# axs[0].set_xlabel('Time (ms)')

# Counter neurons
n_id = 0
for segment in switch_data:
    n_tmp = len(segment.spiketrains)
    for i in range(n_tmp):
        axs[1].plot(segment.spiketrains[i], [n_id] * len(segment.spiketrains[i]), 'o', markersize=0.5,
                    color='darkmagenta')
        n_id += 1

axs[1].set_xlim([0, global_params["sim_time"]])
axs[1].set_ylim([-1, 3 * n_bits])
axs[1].set_yticks(range(0, 3 * n_bits, 3))
# axs[1].set_xlabel('Time (ms)')
axs[1].set_ylabel('Neuron IDs')

# Inputs
axs[2].plot(all_spikes, [0] * len(all_spikes), 'o', markersize=0.5, color='orange')
axs[2].set_xlim([0, global_params["sim_time"]])
axs[2].set_ylim([-1, 1])
axs[2].set_yticks([0], labels=[""])
axs[2].set_xlabel('Time (ms)')
axs[2].set_ylabel('Input')

plt.tight_layout()
plt.savefig("experiments/" + filename + '.png', transparent=False, facecolor='white', edgecolor='black')
plt.show()