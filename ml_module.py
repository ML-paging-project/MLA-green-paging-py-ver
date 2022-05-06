import math
import random
from collections import namedtuple, deque, OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

from dqn import DQN
from lru import run_lru


def get_vector(seq, pointer_position, window_size):
    '''
    pointer_position show which request in the seq we want to build feature.
    window_size: how many requests we look back to build the feature.
    Output is a list of length window_size+4.
    The i th number in the list (1<=i<=window_size) is the frequency of
    the i th most frequent id.
    Chop the window into 4 segments, the last 4 number show
    how many distinct page ids in each seg
    '''
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
        segs_distinct = [0.00, 0.00, 0.00, 0.00]  # chop the window into 4 segments
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


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Maverick: I add a no_more_random.
# In the final epochs of training, I can choose to ban the model from acting randomly.
# Guarantee the model converge.
def select_action(policy_net, state, steps_done, EPS_START, EPS_END,
                  EPS_DECAY, no_more_random, num_of_box_kinds):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold or no_more_random:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return torch.argmax(policy_net(state)), steps_done
    else:
        return torch.tensor([[random.randrange(num_of_box_kinds)]],
                            device=device, dtype=torch.long), steps_done


def optimize_model(memory, BATCH_SIZE, ALPHA, GAMMA,
                   policy_net, target_net, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(memory) < BATCH_SIZE:
        return policy_net, optimizer
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # print(state_batch.shape)
    state_action_values = policy_net(state_batch).gather(0, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    nsv = target_net(non_final_next_states).max(1)[0].detach()
    lst = []
    for row in nsv:
        lst.append(torch.argmax(row) * 1.00)
    next_state_values[non_final_mask] = torch.tensor(lst, device=device)
    # Compute the expected Q values

    ##############################################
    ## Maverick: I add learning rate INIT_ALPHA here ##
    ##############################################
    expected_state_action_values = ALPHA * ((next_state_values * GAMMA) + reward_batch)

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return policy_net, optimizer


def train_model(k, BATCH_SIZE, INIT_ALPHA, GAMMA, EPS_START, EPS_END,
                EPS_DECAY, TARGET_UPDATE, window_size,
                miss_cost, number_of_box_kinds, NUMBER_OF_MODELS,
                num_episodes, seq):
    print("begin to train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features = []
    for i in range(len(seq)):
        features.append([[get_vector(seq, i, window_size)]])
    features = torch.tensor(features, device=device)

    best_result = [math.inf for _ in range(num_episodes)]
    best_hist = [0 for _ in range(number_of_box_kinds)]
    print(len(best_result))
    best_policy_net = DQN(window_size, number_of_box_kinds).to(device)
    best_target_net = DQN(window_size, number_of_box_kinds).to(device)

    for model in range(NUMBER_OF_MODELS):
        alpha = INIT_ALPHA
        decay = (alpha - 0.1) / num_episodes
        global_lru = OrderedDict()
        steps_done = 0
        policy_net = DQN(window_size, number_of_box_kinds).to(device)
        target_net = DQN(window_size, number_of_box_kinds).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = optim.RMSprop(policy_net.parameters())
        memory = ReplayMemory(10000)
        result = []
        hist = [0 for _ in range(number_of_box_kinds)]

        for i_episode in range(num_episodes):
            for xx in range(number_of_box_kinds):
                hist[xx] = 0
            pointer = 0
            impact = 0
            while pointer < len(seq):
                startpointer = pointer
                state = features[pointer]
                # state = torch.tensor([[state]], device=device)
                state = torch.transpose(state, 1, 2)
                action, steps_done = select_action(policy_net, state, steps_done,
                                                   EPS_START, EPS_END, EPS_DECAY,
                                                   False,
                                                   number_of_box_kinds)
                box_id = action
                # print(box_id)
                hist[box_id] = hist[box_id] + 1
                cache_size = k/(2 ** box_id)
                box_width = miss_cost * cache_size

                # Compartmentalization
                # Load top pages from LRU stack.
                mycache = OrderedDict()
                for pid in global_lru.keys():
                    mycache[pid] = True
                    mycache.move_to_end(pid, last=True)  ###########
                    if len(mycache) == cache_size:
                        break

                action = torch.tensor([[[action]]], device=device)  # make it a tensor
                pointer = run_lru(mycache, cache_size, seq, pointer, box_width, 1, miss_cost)
                endpointer = pointer

                # Update global stack
                for x in range(startpointer, endpointer):
                    if seq[x] in global_lru.keys():
                        global_lru.move_to_end(seq[x], last=False)
                    else:
                        global_lru[seq[x]] = True
                        global_lru.move_to_end(seq[x], last=False)

                # print(mi)
                area = 3 * miss_cost * cache_size * cache_size
                impact = impact + area
                reward = torch.tensor([[[-area]]], device=device)

                if pointer < len(seq):
                    next_state = features[pointer]
                    # next_state = torch.tensor([[next_state]], device=device)
                    next_state = torch.transpose(next_state, 1, 2)
                else:
                    next_state = None

                # Store the transition in memory
                memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                policy_net, optimizer = optimize_model(memory, BATCH_SIZE, alpha, GAMMA,
                                                       policy_net, target_net, optimizer)

            # Update the target network, copying all weights and biases in DQN
            # clear_output()
            print('still training')
            print(best_result[-1])
            print('MODEL-' + str(model))
            print('epoch=' + str(i_episode) + '..........impact=' + str(impact))
            result.append(impact.item())
            print(result)
            alpha -= decay
            #alpha=alpha/5

            if i_episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        print('Complete')
        if result[-1] < best_result[-1]:
            for idx in range(len(result)):
                best_result[idx] = result[idx]
            for idx in range(len(hist)):
                best_hist[idx] = hist[idx]

            # store the best for further use
            best_policy_net.load_state_dict(policy_net.state_dict())
            best_target_net.load_state_dict(target_net.state_dict())

    print(best_hist)
    # torch.save({'parameters': best_policy_net.state_dict()}, 'best_cactusadm_train.t7')
    return best_policy_net, best_result
