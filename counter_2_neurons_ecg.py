import os
import pickle

import numpy as np
import spynnaker8 as sim
import wfdb
from matplotlib import pyplot as plt

# Neuron parameters
global_params = {"min_delay": 1.0, "sim_time": 150.0}
neuron_params = {"cm": 0.1, "tau_m": 0.1, "tau_refrac": 0.0, "tau_syn_E": 0.1, "tau_syn_I": 0.1, "v_rest": -65.0, "v_reset": -65.0, "v_thresh": -64.91}
oc_params = {"cm": 1.0, "tau_m": 10.0, "tau_refrac": 75.0, "tau_syn_E": 40.0, "v_rest": -65.0, "v_reset": -65.0, "v_thresh": -43.0}  # 1 QRS each 150 ms if f = 400 pulse/min

if __name__ == '__main__':
    # Read sample
    n_secs = 3
    global_params["sim_time"] = float(n_secs * 1000)
    freq = 360

    # Comprobación
    '''record = wfdb.rdrecord('data/100', sampto=n_secs * freq)
    ann = wfdb.rdann('data/100', 'atr', sampto=n_secs * freq)
    wfdb.plot_wfdb(record, annotation=ann)'''

    # Carga de los datos
    signals, fields = wfdb.rdsamp('data/117', sampto=int(n_secs * freq))

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

    # Secs to milisecs
    on_spikes = np.array(on_spikes) * 1000
    off_spikes = np.array(off_spikes) * 1000
    all_spikes = np.sort(np.concatenate((on_spikes, off_spikes)))

    # --- Simulation ---
    sim.setup(global_params["min_delay"])

    # --- Predefined objects ---
    std_conn = sim.StaticSynapse(weight=1.0, delay=global_params["min_delay"])  # Standard connection
    n_bits = 2

    # -- Network architecture --
    # - Spike injectors -
    src_count = sim.Population(1, sim.SpikeSourceArray(spike_times=all_spikes))

    # - Populations -
    switch_array = []
    and_array = []

    for i in range(n_bits):
        switch_pop = sim.Population(3, sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]}, label="switch")
        and_pop = sim.Population(3, sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]}, label="and")
        switch_array.append(switch_pop)
        and_array.append(and_pop)

    oc_pop_v1 = sim.Population(1, sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]}, label="oc")  # Overflow count
    oc_pop_v2 = sim.Population(1, sim.IF_curr_exp(**oc_params), initial_values={'v': oc_params["v_rest"]}, label="oc")  # Overflow count

    # - Connections -
    # Count signal (Bit 0) - Switch
    sim.Projection(src_count, sim.PopulationView(switch_array[0], [0]), sim.OneToOneConnector(), std_conn)
    sim.Projection(src_count, sim.PopulationView(switch_array[0], [1, 2]), sim.AllToAllConnector(), std_conn, receptor_type="inhibitory")

    # Count signal (Bit 0) - AND
    sim.Projection(src_count, sim.PopulationView(and_array[0], [0]), sim.OneToOneConnector(), std_conn)
    sim.Projection(src_count, sim.PopulationView(and_array[0], [2]), sim.OneToOneConnector(), std_conn)
    sim.Projection(sim.PopulationView(and_array[0], [2]), sim.PopulationView(and_array[0], [1]), sim.OneToOneConnector(), std_conn)

    # Internal (Switch)
    for i in range(n_bits):
        # Interconnections
        sim.Projection(sim.PopulationView(switch_array[i], [0]), sim.PopulationView(switch_array[i], [1]), sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(switch_array[i], [1, 2]), sim.PopulationView(switch_array[i], [0]), sim.AllToAllConnector(), std_conn, receptor_type="inhibitory")

        # Recurrence
        sim.Projection(sim.PopulationView(switch_array[i], [0]), sim.PopulationView(switch_array[i], [0]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
        sim.Projection(sim.PopulationView(switch_array[i], [1]), sim.PopulationView(switch_array[i], [2]), sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(switch_array[i], [2]), sim.PopulationView(switch_array[i], [1]), sim.OneToOneConnector(), std_conn)

    # AND
    for i in range(n_bits):
        sim.Projection(switch_array[i], sim.PopulationView(and_array[i], [0]), sim.AllToAllConnector(), std_conn, receptor_type="inhibitory")
        sim.Projection(sim.PopulationView(and_array[i], [0]), sim.PopulationView(and_array[i], [1]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")

        # Next AND
        if i < n_bits - 1:
            sim.Projection(sim.PopulationView(and_array[i], [1]), sim.PopulationView(and_array[i+1], [0]), sim.OneToOneConnector(), std_conn)
            sim.Projection(sim.PopulationView(and_array[i], [1]), sim.PopulationView(and_array[i+1], [2]), sim.OneToOneConnector(), std_conn)
            sim.Projection(sim.PopulationView(and_array[i+1], [2]), sim.PopulationView(and_array[i+1], [1]), sim.OneToOneConnector(), std_conn)

            # Next Switch
            sim.Projection(sim.PopulationView(and_array[i], [1]), sim.PopulationView(switch_array[i + 1], [0]), sim.OneToOneConnector(), std_conn)
            sim.Projection(sim.PopulationView(and_array[i], [1]), sim.PopulationView(switch_array[i + 1], [1, 2]), sim.AllToAllConnector(), std_conn, receptor_type="inhibitory")

    # ECG COUNTING OVERFLOW
    sim.Projection(sim.PopulationView(and_array[-1], [1]), oc_pop_v1, sim.OneToOneConnector(), std_conn)
    sim.Projection(sim.PopulationView(and_array[-1], [1]), oc_pop_v2, sim.OneToOneConnector(), std_conn)

    # -- Recording --
    for i in range(n_bits):
        switch_array[i].record(["spikes"])
    for i in range(n_bits - 1):
        and_array[i].record(["spikes"])
    oc_pop_v1.record(["spikes"])
    oc_pop_v2.record(["spikes"])

    # -- Run simulation --
    sim.run(global_params["sim_time"])

    # -- Get data from the simulation --
    switch_data = [switch_array[i].get_data().segments[0] for i in range(n_bits)]
    and_data = [and_array[i].get_data().segments[0] for i in range(n_bits - 1)]
    oc_data = [oc_pop_v1.get_data().segments[0], oc_pop_v2.get_data().segments[0]]

    # - End simulation -
    sim.end()

    # --- Saving test ---
    save_array = [switch_data, and_data, all_spikes, oc_data]
    test_name = os.path.basename(__file__).split('.')[0]

    cwd = os.getcwd()
    if not os.path.exists(cwd + "/experiments/"):
        os.mkdir(cwd + "/experiments/")

    i = 1
    while os.path.exists(cwd + "/experiments/" + test_name + "_" + str(i) + ".pickle"):
        i += 1

    filename = test_name + "_" + str(i)

    with open("experiments/" + filename + '.pickle', 'wb') as handle:
        pickle.dump(save_array, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # --- Saving plot ---
    plt.rcParams['figure.dpi'] = 400
    plt.rcParams['font.size'] = '4'
    plt.rcParams["figure.figsize"] = (4, 1.5)

    fig, axs = plt.subplots(3, 1, gridspec_kw={'height_ratios': [2, 6, 1]}, sharex=True)
    #fig.suptitle('Spiking response')

    # Overflow counter
    axs[0].plot(oc_data[0].spiketrains[0], [0] * len(oc_data[0].spiketrains[0]), 'o', markersize=0.5, color='darkmagenta')
    axs[0].plot(oc_data[1].spiketrains[0], [1] * len(oc_data[1].spiketrains[0]), 'o', markersize=0.5, color='palevioletred')
    axs[0].set_xlim([0, global_params["sim_time"]])
    axs[0].set_ylim([-1, 2])
    axs[0].set_yticks([0, 1], labels=["CO", "FO"])  # Counter output, Filter output
    #axs[0].set_xlabel('Time (ms)')

    # Counter neurons
    n_id = 0
    for segment in switch_data:
        n_tmp = len(segment.spiketrains)
        for i in range(n_tmp):
            axs[1].plot(segment.spiketrains[i], [n_id] * len(segment.spiketrains[i]), 'o', markersize=0.5, color='darkmagenta')
            n_id += 1

    axs[1].set_xlim([0, global_params["sim_time"]])
    axs[1].set_ylim([-1, 3 * n_bits])
    axs[1].set_yticks(range(0, 3 * n_bits, 3))
    #axs[1].set_xlabel('Time (ms)')
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
