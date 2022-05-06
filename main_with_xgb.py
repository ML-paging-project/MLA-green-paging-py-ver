import tool
from xgboost_module import run_xgboost
from other_algorithms import opt, michael, random_pick
import matplotlib.pyplot as plt
import os

files = os.listdir('datasets')
# print(files)
names = {}
for f in files:
    temp = f.split('_')[0]
    names[temp] = 0

print('\n\n')
tick = 1
for trace_name in names:
    # trace_name = 'bzip'
    print('++++++++++ running', tick, '/', len(names), '++++++++++')
    tick += 1
    train_seq = tool.read_crc_seq('datasets/' + trace_name + '_train.csv')
    test_seq = tool.read_crc_seq('datasets/' + trace_name + '_test.csv')

    k = 2 ** 7
    number_of_box_kinds = 8
    window_size = 256
    miss_cost = 1000
    print('########### play with xgb ###########')
    xgb_impact = run_xgboost(train_seq, test_seq, k, number_of_box_kinds,
                             miss_cost, window_size)
    print('######### testing other methods ############')
    opt_impact, _, _ = opt(test_seq, k, number_of_box_kinds, miss_cost)
    michael_impact = michael(test_seq, k, number_of_box_kinds, miss_cost)
    random_impact = random_pick(test_seq, k, number_of_box_kinds, miss_cost)
    print('oracle:', xgb_impact)
    print('opt:', opt_impact)
    print('random:', random_impact)
    print('Michael:', michael_impact)

    h = ['Random', 'Michael', 'Oracle']
    v = [random_impact / opt_impact, michael_impact / opt_impact, xgb_impact / opt_impact]
    plt.bar(h, v, width=0.3, color='red')
    plt.xlabel('methods', fontdict={'weight': 'black'})
    plt.grid(axis='y', linestyle='-.', linewidth=1, color='black', alpha=0.5)
    plt.ylabel('competitive ratio on memory impact', fontdict={'weight': 'black'})
    plt.ylim(0, max(v) * 1.1)
    plt.title('Memory impact on the test trace of ' + trace_name+', s=1k',
              fontdict={'weight': 'black'})
    for index, value in enumerate(v):
        plt.text(index - 0.07, value + 0.1, str(round(value, 2)),
                 fontdict={'weight': 'black'})
    # plt.show()
    plt.savefig(r'plot-s-1000/s1k-' + trace_name + '.jpg')
    plt.close()
