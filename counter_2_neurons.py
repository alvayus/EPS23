import os
import pickle

import spynnaker8 as sim
from matplotlib import pyplot as plt

# Neuron parameters
global_params = {"min_delay": 1.0, "sim_time": 150.0}
neuron_params = {"cm": 0.1, "tau_m": 0.1, "tau_refrac": 0.0, "tau_syn_E": 0.1, "tau_syn_I": 0.1, "v_rest": -65.0, "v_reset": -65.0, "v_thresh": -64.91}


if __name__ == '__main__':
    # --- Simulation ---
    sim.setup(global_params["min_delay"])

    # --- Predefined objects ---
    std_conn = sim.StaticSynapse(weight=1.0, delay=global_params["min_delay"])  # Standard connection
    n_bits = 4

    # -- Network architecture --
    # - Spike injectors -
    count_times = range(10, int(global_params["sim_time"] - 20), 1)
    #count_times = range(10, 15, 1)
    src_count = sim.Population(1, sim.SpikeSourceArray(spike_times=count_times))

    # - Populations -
    switch_array = []
    and_array = []

    for i in range(n_bits):
        switch_pop = sim.Population(3, sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]}, label="switch")
        switch_array.append(switch_pop)

    for i in range(n_bits - 1):
        and_pop = sim.Population(3, sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]}, label="and")
        and_array.append(and_pop)

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
    for i in range(n_bits - 1):
        sim.Projection(switch_array[i], sim.PopulationView(and_array[i], [0]), sim.AllToAllConnector(), std_conn, receptor_type="inhibitory")
        sim.Projection(sim.PopulationView(and_array[i], [0]), sim.PopulationView(and_array[i], [1]), sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")

        # Next AND
        if i < n_bits - 2:
            sim.Projection(sim.PopulationView(and_array[i], [1]), sim.PopulationView(and_array[i+1], [0]), sim.OneToOneConnector(), std_conn)
            sim.Projection(sim.PopulationView(and_array[i], [1]), sim.PopulationView(and_array[i+1], [2]), sim.OneToOneConnector(), std_conn)
            sim.Projection(sim.PopulationView(and_array[i+1], [2]), sim.PopulationView(and_array[i+1], [1]), sim.OneToOneConnector(), std_conn)

        # Next Switch
        sim.Projection(sim.PopulationView(and_array[i], [1]), sim.PopulationView(switch_array[i + 1], [0]), sim.OneToOneConnector(), std_conn)
        sim.Projection(sim.PopulationView(and_array[i], [1]), sim.PopulationView(switch_array[i + 1], [1, 2]), sim.AllToAllConnector(), std_conn, receptor_type="inhibitory")

    # -- Recording --
    for i in range(n_bits):
        switch_array[i].record(["spikes"])
    for i in range(n_bits - 1):
        and_array[i].record(["spikes"])

    # -- Run simulation --
    sim.run(global_params["sim_time"])

    # -- Get data from the simulation --
    switch_data = [switch_array[i].get_data().segments[0] for i in range(n_bits)]
    and_data = [and_array[i].get_data().segments[0] for i in range(n_bits - 1)]

    # - End simulation -
    sim.end()

    # --- Saving test ---
    save_array = [switch_data, and_data, count_times]
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

    print(len(count_times))  # Number of overflows * Counter capacity + Last number

    # --- Saving plot ---
    plt.rcParams['figure.dpi'] = 400
    plt.rcParams['font.size'] = '4'
    plt.rcParams["figure.figsize"] = (4, 1.5)

    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle('Spiking response')

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
    axs[0].set_xlabel('Time (ms)')
    axs[0].set_ylabel('Neuron IDs')

    # Inputs
    axs[1].plot(count_times, [0] * len(count_times), 'o', markersize=0.5, color='orange')
    axs[1].set_xlim([0, global_params["sim_time"]])
    axs[1].set_ylim([-1, 1])
    axs[1].set_yticks([0])
    axs[1].set_xlabel('Time (ms)')
    axs[1].set_ylabel('Spike generator')

    plt.tight_layout()
    plt.savefig("experiments/" + filename + '.png', transparent=False, facecolor='white', edgecolor='black')
    plt.show()
