import other_algorithms as oag
import xgboost as xgb
from lru import run_lru
from collections import OrderedDict


def get_feature_vector(seq, pointer_position, window_size):
    """
    pointer_position show which request in the seq we want to build feature.
    window_size: how many requests we look back to build the feature.
    Output is a list of length window_size+4.
    The i th number in the list (1<=i<=window_size) is the frequency of
    the i th most frequent id.
    Chop the window into 4 segments, the last 4 number show
    how many distinct page ids in each seg
    """
    vector = []
    if pointer_position == 0:
        for i in range(window_size + 4):
            vector.append(0.00)
    else:
        if pointer_position < window_size:
            window = seq[0:pointer_position]
        else:
            window = seq[pointer_position - window_size:pointer_position]
        frequency = {}
        distinct = {}
        segs_distinct = [0.00, 0.00, 0.00, 0.00]
        # chop the window into 4 segments
        # count how many distinct page ids in each seg

        i = 0
        seg_id = 0
        while i < len(window):
            if window[i] in frequency.keys():
                frequency[window[i]] = frequency[window[i]] + 1.00
            else:
                frequency[window[i]] = 1.00
            distinct[window[i]] = True
            if (i + 1) % (int(window_size / 4)) == 0:
                # segment ends. see how many distinct items in this seg.
                segs_distinct[seg_id] = len(distinct.keys())
                distinct = {}
                seg_id = seg_id + 1
            i = i + 1

        if len(distinct.keys()) > 0:
            segs_distinct[seg_id] = len(distinct.keys())

        # 1<=j<=window_size, the j-th variable of the vector is
        # the frequency of the j-th most frequent page id in
        # the window
        for v in frequency.values():
            vector.append(v)
        while len(vector) < window_size:
            vector.append(0)
        vector.sort(reverse=True)

        # We chop the window_size requests into four segments of length w/4.
        # Count how many distinct ids in each segment, and put the countings in the last four variables.
        for v in segs_distinct:
            vector.append(v)
    return vector


def build_train_set(sequence, k, number_of_box_kinds, miss_cost, window_size):
    print('calculating features...')
    features = []
    for i in range(len(sequence)):
        features.append(get_feature_vector(sequence, i, window_size))

    opt_mi, opt_boxes, decision_points = oag.opt(sequence, k, number_of_box_kinds, miss_cost)
    xs = []
    for i in range(len(opt_boxes)):
        xs.append(features[decision_points[i]])
    boxes = {}
    for x in opt_boxes:
        boxes[x] = True
    boxes = sorted(boxes.keys(), reverse=False)
    box2y = {}
    y2box = {}
    for i in range(len(boxes)):
        box2y[boxes[i]] = i
        y2box[i] = boxes[i]
    ys = [box2y[b] for b in opt_boxes]
    return xs, ys, y2box


def run_xgboost(train_seq, test_seq, k, number_of_box_kinds, miss_cost, window_size):
    train_x, train_y, y2box = build_train_set(train_seq, k, number_of_box_kinds,
                                              miss_cost, window_size)
    params = {
        'objective': 'multi:softmax',
        'eta': 0.1,
        'max_depth': 5,
        'num_class': len(y2box.keys())
    }
    print('training xgboost...')
    xgb_train = xgb.DMatrix(train_x, label=train_y)
    num_round = 10
    model = xgb.train(params, xgb_train, num_round)

    print('testing xgboost...')
    pointer = 0
    total_impact = 0
    global_lru = OrderedDict()
    while pointer < len(test_seq):
        start_pointer = pointer
        feature = xgb.DMatrix([get_feature_vector(test_seq, pointer, window_size)])
        oracle = y2box[model.predict(feature)[0]]
        cache_size = k / (2 ** oracle)
        box_width = miss_cost * cache_size
        # Compartmentalization
        # Load top pages from LRU stack.
        mycache = OrderedDict()
        for pid in global_lru.keys():
            mycache[pid] = True
            mycache.move_to_end(pid, last=True)  ###########
            if len(mycache) == cache_size:
                break

        pointer = run_lru(mycache, cache_size, test_seq, pointer, box_width, 1, miss_cost)
        endpointer = pointer

        # Update global stack
        for x in range(start_pointer, endpointer):
            if test_seq[x] in global_lru.keys():
                global_lru.move_to_end(test_seq[x], last=False)
            else:
                global_lru[test_seq[x]] = True
                global_lru.move_to_end(test_seq[x], last=False)

        mi = 3 * miss_cost * cache_size * cache_size
        total_impact = total_impact + mi
    return total_impact
