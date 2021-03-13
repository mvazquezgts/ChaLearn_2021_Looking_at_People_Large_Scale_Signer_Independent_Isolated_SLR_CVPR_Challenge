import random

import numpy as np


def downsample(data_numpy, step, random_sample=True):
    # input: C,T,V,M
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]


def temporal_slice(data_numpy, step):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T / step, step, V, M).transpose(
        (0, 1, 3, 2, 4)).reshape(C, T / step, V, step * M)


def mean_subtractor(data_numpy, mean):
    # input: C,T,V,M
    # naive version
    if mean == 0:
        return
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy


def auto_pading(data_numpy, size, random_pad=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M))
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


def random_choose(data_numpy, size, auto_pad=True):
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]


def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def random_shift(data_numpy):
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift

def flip(data_numpy):
    
    flipped_data = np.copy(data_numpy)
    flipped_data[0,:,:,:] *= -1
    
    return flipped_data


def random_flip(data_numpy):
    
    if (np.random.rand()>0.5):
        return flip(data_numpy)
    
    return data_numpy


def dropout(data_numpy):
    
    threshold = 0.1
    k_prob = np.array([0.914753049951041,0.8594778493712205,0.7729485879554081,0.7114091151572326,0.7922072385186532,0.7760843978733091,0.73422800841149,0.8333623249697306,0.6276307390724284,0.9241670870055689,0.8746154987000795,0.8505170697933112,0.8530597700896883,0.5877752176741085,0.5689137068429606,0.8259425959118061,0.7887293913661346,0.8681423253473648,0.6805635297370227,0.7758115887447357,0.7743001077557887,0.8144174028047132,0.7247338184931961,0.6920247647752444,0.7638070246615507,0.7370704846378939,0.7564925859742406,0.707821938399865,0.6170845977764099,0.5046231497725013,0.7355631603758805,0.6888687019227212,0.676419959093343,0.48224235662168763,0.5543929709866354,0.5932907164668286,0.8238087724112451,0.7709895437848826,0.8481683473532052,0.663909158541673,0.7949270259061817,0.6975233913385155,0.8421050974651694,0.748720313478265,0.7413259734894218,0.7519308611956179,0.8799407207869778,0.6910604749023914,0.7211768744013414,0.7337947077032851,0.830611656527817,0.6535854002824113,0.7702615262847599,0.8272240700516436,0.8659898282497556])
    
    data = np.copy(data_numpy)
    kp_dropout = np.flatnonzero(((k_prob * np.random.rand(len(k_prob))) < threshold).astype(int))
    
    data[:,:,kp_dropout,:] *= 0
    
    return data

def resizer(data_numpy, resize):
    data = np.copy(data_numpy)
    data *= resize
    return data

def random_resizer(data_numpy):
    random_size = np.random.uniform(low=0.9, high=1.1)
    return resizer(data_numpy, random_size)

def use_tta(data_numpy, tta):
    data = np.copy(data_numpy)
    if (tta[0] == True):
        data = flip(data)
    data = resizer(data, tta[1])
    return data
    

def openpose_match(data_numpy):
    C, T, V, M = data_numpy.shape
    assert (C == 3)
    score = data_numpy[2, :, :, :].sum(axis=1)
    # the rank of body confidence in each frame (shape: T-1, M)
    rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1, M)

    # data of frame 1
    xy1 = data_numpy[0:2, 0:T - 1, :, :].reshape(2, T - 1, V, M, 1)
    # data of frame 2
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    # square of distance between frame 1&2 (shape: T-1, M, M)
    distance = ((xy2 - xy1) ** 2).sum(axis=2).sum(axis=0)

    # match pose
    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = (rank == m)
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert (np.all(forward_map >= 0))

    # string data
    for t in range(T - 1):
        forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

    # generate data
    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[
                                                             t]].transpose(1, 2, 0)
    data_numpy = new_data_numpy

    # score sort
    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:, :, :, rank]

    return data_numpy
