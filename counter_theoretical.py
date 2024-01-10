import os

from matplotlib import pyplot as plt

global_params = {"min_delay": 1.0, "sim_time": 150.0}

if __name__ == '__main__':
    n_bits = 4
    count_times = range(10, int(global_params["sim_time"] - 20), 1)

    # --- Saving test ---
    test_name = os.path.basename(__file__).split('.')[0]

    cwd = os.getcwd()
    if not os.path.exists(cwd + "/experiments/"):
        os.mkdir(cwd + "/experiments/")

    i = 1
    while os.path.exists(cwd + "/experiments/" + test_name + "_" + str(i) + ".pickle"):
        i += 1

    filename = test_name + "_" + str(i)

    n_count = len(count_times)
    print(n_count)  # Number of overflows * Counter capacity + Last number

    # --- Saving plot ---
    plt.rcParams['figure.dpi'] = 400
    plt.rcParams['font.size'] = '4'
    plt.rcParams["figure.figsize"] = (4, 0.8)

    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    #fig.suptitle('Spiking response')

    # Spikes
    times = [[] for i in range(n_bits)]

    count = 0
    count_bin = None
    for t in count_times:
        count = (count + 1) % (2 ** n_bits)
        count_bin = ("{:0" + str(n_bits) + "b}").format(count)
        print(count_bin)

        for j in range(len(count_bin)):
            if count_bin[j] == '1':
                times[4 - j - 1].append(t)

    for i in range(count_times[-1], int(global_params["sim_time"])):
        for j in range(len(count_bin)):
            if count_bin[j] == '1':
                times[4 - j - 1].append(i)

    for i in range(len(times)):
        axs[0].plot(times[i], [i] * len(times[i]), 'o', markersize=0.5, color='orange')

    axs[0].set_xlim([0, global_params["sim_time"]])
    axs[0].set_ylim([-0.5, n_bits - 0.5])
    axs[0].set_yticks(range(0, n_bits))
    #axs[0].set_xlabel('Time (ms)')
    axs[0].set_ylabel('Bit')

    # Inputs
    axs[1].plot(count_times, [0] * len(count_times), 'o', markersize=0.5, color='orange')
    axs[1].set_xlim([0, global_params["sim_time"]])
    axs[1].set_ylim([-0.5, 0.5])
    axs[1].set_yticks([0], labels=[" "])
    axs[1].set_xlabel('Time (ms)')
    axs[1].set_ylabel('Input')

    plt.tight_layout()
    plt.savefig("experiments/" + filename + '.png', transparent=False, facecolor='white', edgecolor='black')
    plt.show()
