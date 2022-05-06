def read_crc_seq(file):
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
