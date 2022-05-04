import matplotlib.pyplot as plt

import other_algorithms as alg
from ml_module import train_model


# You can use this function to read CRC trace
def read_seq(file):
    sequence = []
    with open(file, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            if len(line) == 0:
                continue
            data = line.split(',')
            sequence.append(data[0])
    f.close()
    return sequence


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # read data
    f = open("seq-sort10k.ssv")
    data = f.readline()
    seq = data.split(' ')
    print(seq[:10])
    print(len(seq))

    # to use CRC traces:
    # seq = read_seq('cactusadm_train.csv')
    # print(seq[:10])
    # print(len(seq))
    # CRC traces: https://github.com/chledowski/Robust-Learning-Augmented-Caching-An-Experimental-Study-Datasets/tree/main/datasets

    # ML parameters
    # see https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    BATCH_SIZE = 8
    GAMMA = 0.3
    EPS_START = 0.9
    EPS_END = 0.00001
    EPS_DECAY = 500
    TARGET_UPDATE = 5
    window_size = 256  # for features
    miss_cost = 1000  ##################
    # number_of_box_kinds = min(8,math.ceil(math.log2(miss_cost))) #############
    number_of_box_kinds = 8
    k = 2 ** 7
    NUMBER_OF_MODELS = 10  # Train several models, choose the best
    num_episodes = 20
    ALPHA = 0.8  # Learning rate #########################

    net, best_result = train_model(BATCH_SIZE, ALPHA, GAMMA, EPS_START, EPS_END,
                                   EPS_DECAY, TARGET_UPDATE, window_size,
                                   miss_cost, number_of_box_kinds, NUMBER_OF_MODELS,
                                   num_episodes, seq)
    opt_impact, _ = alg.opt(seq, k, number_of_box_kinds, miss_cost)
    michael_impact = alg.michael(seq, k, number_of_box_kinds, miss_cost)
    random_impact = alg.random_pick(seq, k, number_of_box_kinds, miss_cost)

    # draw plots
    plt.figure(figsize=(16, 9), dpi=50)
    plt.xlabel('Training epoch', fontsize=20)
    plt.ylabel('Memory impact', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(0, max(random_impact, max(best_result), michael_impact) + 500000000)
    plt.plot([x for x in range(1, num_episodes + 1)], best_result, label='oracle',
             linestyle='-', color='g', marker='x', linewidth=1.5)
    plt.plot([x for x in range(1, num_episodes + 1)],
             [random_impact for _ in range(num_episodes)], label='random',
             linestyle='-', color='k', marker='x', linewidth=1.5)
    plt.plot([x for x in range(1, num_episodes + 1)],
             [michael_impact for _ in range(num_episodes)], label='Michael',
             linestyle='-', color='b', marker='x', linewidth=1.5)
    plt.plot([x for x in range(1, num_episodes + 1)],
             [opt_impact for _ in range(num_episodes)], label='offline opt',
             linestyle='-', color='r', marker='x', linewidth=1.5)
    plt.ticklabel_format(style='plain', axis='both')
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
    plt.legend(fontsize=20)

    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
