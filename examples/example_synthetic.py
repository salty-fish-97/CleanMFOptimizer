from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, CategoricalHyperparameter, \
    UniformIntegerHyperparameter
import time
import numpy as np

cs = ConfigurationSpace()
num_num = 16
num_cat = 8
rng = np.random.RandomState(1)

for i in range(num_num):
    hp = UniformFloatHyperparameter('num_%s' % i, 0, 1)
    cs.add_hyperparameter(hp)
for i in range(num_cat):
    hp = CategoricalHyperparameter('cat_%s' % i, [0, 1])
    cs.add_hyperparameter(hp)


def test_func(config, resource_ratio):
    # print(dict(config))
    max_sampling_times = 243
    x_cat = [config[key] for key in config if 'cat' in key]
    x_num = [config[key] for key in config if 'num' in key]

    result = -np.sum(x_cat)
    # draw samples to approximate the expectation (Bernoulli distribution)
    n_samples = int(max_sampling_times * resource_ratio)
    for x in x_num:
        result -= rng.binomial(n_samples, p=x) / n_samples

    # Remember, we minimize the return value
    return result


import os
import sys

sys.path.append(os.getcwd())
rep = 30

from openbox import Advisor, Observation

budget = 30

smac_record_dict_list = []

for _ in range(rep):
    time_cnt = 0
    start_time = time.time()
    smac_record_dict = {}
    advisor = Advisor(config_space=cs,
                      surrogate_type='prf',
                      acq_type='ei',
                      acq_optimizer_type='local_random')
    while time.time() - start_time < budget:
        config = advisor.get_suggestion()
        time_cnt += 27
        result = test_func(config, resource_ratio=27)
        smac_record_dict[time_cnt] = result
        observation = Observation(config=config, objectives=[result])
        advisor.update_observation(observation)
    smac_record_dict_list.append(smac_record_dict)

from bohb import BOHBOptimizer
# from hyperband import HyperbandOptimizer
from mfes import MFESOptimizer

budget = 15
bohb_record_dict_list = []

for _ in range(rep):
    start_time = time.time()
    bohb_record_dict = {}
    time_cnt = 0
    optimizer = BOHBOptimizer(config_space=cs)
    while time.time() - start_time < budget:
        configs, resource_ratio = optimizer.get_suggestions()
        ret_dict = {}
        for x in configs:
            time_cnt += resource_ratio * 27
            result = test_func(x, resource_ratio)
            ret_dict[x] = result
            bohb_record_dict[time_cnt] = result
        optimizer.update_observations(ret_dict)
    bohb_record_dict_list.append(bohb_record_dict)

budget = 15
mfes_record_dict_list = []

for _ in range(rep):
    start_time = time.time()
    mfes_record_dict = {}
    time_cnt = 0
    optimizer = MFESOptimizer(config_space=cs)

    while time.time() - start_time < budget:
        configs, resource_ratio = optimizer.get_suggestions()
        ret_dict = {}
        for x in configs:
            time_cnt += resource_ratio * 27
            result = test_func(x, resource_ratio)
            ret_dict[x] = result
            mfes_record_dict[time_cnt] = result
        optimizer.update_observations(ret_dict)
    mfes_record_dict_list.append(mfes_record_dict)


def non_increasing_sequence(dictionary):
    min_result = 1e5
    result_array = []
    result_idx = 0
    for idx in dictionary:
        while result_idx < idx and result_idx < 1000:
            result_array.append(min_result)
            result_idx += 1
        if dictionary[idx] < min_result:
            min_result = dictionary[idx]
    return result_array


from matplotlib import pyplot as plt

x = list(range(1000))
smac_y = np.mean([np.array(non_increasing_sequence(d)) for d in smac_record_dict_list], axis=0)
bohb_y = np.mean([np.array(non_increasing_sequence(d)) for d in bohb_record_dict_list], axis=0)
mfes_y = np.mean([np.array(non_increasing_sequence(d)) for d in mfes_record_dict_list], axis=0)

plt.plot(x, smac_y, label='SMAC')
plt.plot(x, bohb_y, label='BOHB')
plt.plot(x, mfes_y, label='MFES')

print(smac_y)
print(bohb_y)
print(mfes_y)

plt.xlim(0, 1000)
plt.ylim(-18, -13)
plt.legend()
plt.savefig('plot.pdf')
