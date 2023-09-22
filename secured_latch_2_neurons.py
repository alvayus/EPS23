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

    # -- Network architecture --
    # - Populations -
    set_times = [10.0, 40.0, 51.0, 80.0, 81.0, 120.0]
    reset_times = [20.0, 60.0, 100.0]
    src_set = sim.Population(1, sim.SpikeSourceArray(spike_times=set_times))
    src_reset = sim.Population(1, sim.SpikeSourceArray(spike_times=reset_times))

    and_pop = sim.Population(1, sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]}, label="and")
    delay_pop = sim.Population(1, sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]}, label="delay")
    latch_pop = sim.Population(2, sim.IF_curr_exp(**neuron_params), initial_values={'v': neuron_params["v_rest"]}, label="latch")

    # - Connections -
    # Inputs
    sim.Projection(src_set, and_pop, sim.OneToOneConnector(), std_conn)
    sim.Projection(src_reset, delay_pop, sim.OneToOneConnector(), std_conn)

    # Interconnections
    sim.Projection(and_pop, and_pop, sim.OneToOneConnector(), std_conn, receptor_type="inhibitory")
    sim.Projection(and_pop, sim.PopulationView(latch_pop, [0]), sim.OneToOneConnector(), std_conn)
    sim.Projection(delay_pop, latch_pop, sim.AllToAllConnector(), std_conn, receptor_type="inhibitory")
    sim.Projection(latch_pop, and_pop, sim.AllToAllConnector(), std_conn, receptor_type="inhibitory")

    # Recurrence
    sim.Projection(sim.PopulationView(latch_pop, [0]), sim.PopulationView(latch_pop, [1]), sim.OneToOneConnector(), std_conn)
    sim.Projection(sim.PopulationView(latch_pop, [1]), sim.PopulationView(latch_pop, [0]), sim.OneToOneConnector(), std_conn)

    # -- Recording --
    latch_pop.record(["spikes", "v"])

    # -- Run simulation --
    sim.run(global_params["sim_time"])

    # -- Get data from the simulation --
    output_data = latch_pop.get_data().segments[0]

    # - End simulation -
    sim.end()

    # --- Saving test ---
    save_array = [output_data]
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

    fig, axs = plt.subplots(2, 1)
    fig.suptitle('Spiking response')

    # Spikes
    n = len(output_data.spiketrains)
    for i in range(n):
        axs[0].plot(output_data.spiketrains[i], [i] * len(output_data.spiketrains[i]), 'o', markersize=0.5, color='darkmagenta')
    axs[0].set_xlim([0, global_params["sim_time"]])
    axs[0].set_ylim([-1, n])
    axs[0].set_yticks(range(n))
    axs[0].set_xlabel('Time (ms)')
    axs[0].set_ylabel('Neuron IDs')

    # Inputs
    axs[1].plot(set_times, [1] * len(set_times), 'o', markersize=0.5, color='teal')
    axs[1].plot(reset_times, [0] * len(reset_times), 'o', markersize=0.5, color='indianred')
    axs[1].plot([51.0, 81.0], [1] * 2, 'o', markersize=0.5, color='sandybrown')
    axs[1].set_xlim([0, global_params["sim_time"]])
    axs[1].set_ylim([-1, 2])
    axs[1].set_yticks([0, 1], ["Set", "Reset"])
    axs[1].set_xlabel('Time (ms)')
    axs[1].set_ylabel('Spike generators')

    plt.tight_layout()
    plt.savefig("experiments/" + filename + '.png', transparent=False, facecolor='white', edgecolor='black')
    plt.show()