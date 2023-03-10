from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, CategoricalHyperparameter, \
    UniformIntegerHyperparameter
import time

cs = ConfigurationSpace()
hp1 = UniformFloatHyperparameter('x1', 0, 1)
hp2 = UniformFloatHyperparameter('x2', 0, 1)
hp3 = CategoricalHyperparameter('x3', ['a', 'b', 'c'])
hp4 = UniformIntegerHyperparameter('x4', 1, 11, q=2)  # 1,3,5,7,9,11
cs.add_hyperparameters([hp1, hp2, hp3, hp4])


def test_func(config, resource_ratio):
    print(dict(config))

    # Remember, we minimize the return value
    return config['x1'] * resource_ratio


import os
import sys

sys.path.append(os.getcwd())
from bohb import BOHBOptimizer
from hyperband import HyperbandOptimizer
from mfes import MFESOptimizer

optimizer = MFESOptimizer(config_space=cs)

budget = 10
start_time = time.time()

while time.time() - start_time < budget:
    configs, resource_ratio = optimizer.get_suggestions()
    ret_dict = {x: test_func(x, resource_ratio) for x in configs}
    optimizer.update_observations(ret_dict)

print(optimizer.incumbent_config)
print(optimizer.configs)
print(optimizer.perfs)
