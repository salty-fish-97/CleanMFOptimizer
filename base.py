import time
import numpy as np
from math import log, ceil


class BaseMFOptimizer(object):
    def __init__(self, config_space, seed=1, R=27, eta=3, n_jobs=1):
        self.config_space = config_space
        self.n_workers = n_jobs
        self.name = None

        self.trial_cnt = 0
        self.configs = list()
        self.perfs = list()
        self.incumbent_perf = float("-INF")
        self.incumbent_config = self.config_space.get_default_configuration()
        self.incumbent_configs = list()
        self.incumbent_perfs = list()
        self.global_start_time = time.time()

        # Parameters in Hyperband framework.
        self.restart_needed = True
        self.R = R
        self.eta = eta
        self.seed = seed
        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.R))
        self.B = (self.s_max + 1) * self.R
        self.s_values = list(reversed(range(self.s_max + 1)))
        self.inner_iter_id = 0
        self.n_resource = 0

        # Parameters in BOHB.
        self.iterate_r = list()
        self.target_x = dict()
        self.target_y = dict()
        self.exp_output = dict()
        for index, item in enumerate(np.logspace(0, self.s_max, self.s_max + 1, base=self.eta)):
            r = int(item)
            self.iterate_r.append(r)
            self.target_x[r] = list()
            self.target_y[r] = list()

        self.mf_advisor = None

        self.eval_dict = dict()
        self.T = []
        self.tmp_history_dict = dict()
        self.val_losses = []
        self.s = self.s_values[0]
        self.inner_loop_cnt = 0

    def get_suggestions(self, skip_last=0):
        # Set initial number of configurations
        n = int(ceil(self.B / self.R / (self.s + 1) * self.eta ** self.s))
        # initial number of iterations per config
        r = int(self.R * self.eta ** (-self.s))

        if self.inner_loop_cnt == 0:
            print("Suggest a new batch of configurations for the new inner loop.")
            # Suggest a new batch of configurations.
            start_time = time.time()
            self.T = self.mf_advisor.get_suggestions(n)
            time_elapsed = time.time() - start_time
            print("Choosing next batch of configurations took %.2f sec." % time_elapsed)
        else:
            print("Evaluate 1/%d of the previous configurations" % self.eta)
            # Select the top configurations.
            indices = np.argsort(self.val_losses)
            if len(self.T) >= self.eta:
                self.T = [self.T[i] for i in indices]
                reduced_num = int(len(self.T) / self.eta)
                self.T = self.T[0:reduced_num]
            else:
                self.T = [self.T[indices[0]]]

        self.n_resource = r * self.eta ** self.inner_loop_cnt

        # In case the optimizer suggests the same configuration
        self.T = list(set(self.T))
        print("%s: %d configurations x size %d / %d each" %
              (self.name, len(self.T), self.n_resource, self.R))

        self.inner_loop_cnt = (self.inner_loop_cnt + 1) % (self.s + 1)
        if self.inner_loop_cnt == 0:
            self.s = (self.s - 1) % (self.s_max + 1)

        return self.T, self.n_resource / self.R

    def update_observations(self, history_dict):
        raise NotImplementedError
