import pickle
import matplotlib.pyplot as plt

global_params = {"min_delay": 1.0, "sim_time": 3000.0}
n_bits = 2

filename = "counter_2_neurons_ecg_2"
with open("experiments/" + filename + '.pickle', 'rb') as f:
    load_array = pickle.load(f)

switch_data = load_array[0]
and_data = load_array[1]
all_spikes = load_array[2]
oc_data = load_array[3]

# --- Saving plot ---
plt.rcParams['figure.dpi'] = 400
plt.rcParams['font.size'] = '4'
plt.rcParams["figure.figsize"] = (4, 0.7)

fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
# fig.suptitle('Spiking response')

# Overflow counter
axs[0].plot(oc_data[0].spiketrains[0] / 1000, [0] * len(oc_data[0].spiketrains[0]), 'o', markersize=0.5, color='darkmagenta')
axs[0].plot(oc_data[1].spiketrains[0] / 1000, [1] * len(oc_data[1].spiketrains[0]), 'o', markersize=0.5, color='palevioletred')
axs[0].set_xlim([0, global_params["sim_time"]])
axs[0].set_ylim([-1, 2])
axs[0].set_yticks([0, 1], labels=["CO", "FO"])  # Counter output, Filter output
# axs[0].set_xlabel('Time (ms)')

# Inputs
all_spikes = all_spikes / 1000
axs[1].plot(all_spikes, [0] * len(all_spikes), 'o', markersize=0.5, color='orange')
axs[1].set_xlim([0, global_params["sim_time"] / 1000])
axs[1].set_ylim([-1, 1])
axs[1].set_yticks([0], labels=[""])
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Input')

plt.tight_layout()
plt.savefig("experiments/" + filename + '.png', transparent=False, facecolor='white', edgecolor='black')
plt.show()