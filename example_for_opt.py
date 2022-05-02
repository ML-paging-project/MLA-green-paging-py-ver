from other_algorithms import opt

import matplotlib.pyplot as plt


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


seq = read_seq('cactusadm_test.csv')

k = 2 ** 7
number_of_box_kinds = 8
miss_cost = 1000
opt_impact, opt_path = opt(seq, k, number_of_box_kinds, miss_cost)
print('opt_impact=', opt_impact)
print('opt box sequence is', opt_path)

# draw box profile
x = [0]
y = [0]
t = 0
for v in opt_path:
    if v in ['start', 'end']:
        continue
    box_id = int(v.split('-')[0])
    box_size = k / (2 ** box_id)
    x.append(t)
    y.append(box_size)
    t += box_size
    x.append(t)
    y.append(box_size)
x.append(t)
y.append(0)


plt.plot(x, y, linewidth=1, color="orange")
plt.xlabel('time (normalized)')
plt.ylabel('memory size')
plt.show()
