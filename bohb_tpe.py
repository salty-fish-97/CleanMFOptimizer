import time
import numpy as np
from math import log, ceil

from openbox import Observation
from openbox.core.tpe_advisor import TPE_Advisor

from base import BaseMFOptimizer


class BOHBOptimizer_tpe(BaseMFOptimizer):
    def __init__(self, config_space, mode='smac', seed=1, R=27, eta=3, n_jobs=1):
        super().__init__(config_space=config_space,
                         seed=seed, R=R,
                         eta=eta, n_jobs=n_jobs)

        self.mode = mode
        self.name = 'BOHB_TPE'
        self.mf_advisor = TPE_Advisor(config_space=config_space)

    def get_suggestions(self, skip_last=0):
        # Set initial number of configurations
        n = int(ceil(self.B / self.R / (self.s + 1) * self.eta ** self.s))
        # initial number of iterations per config
        r = int(self.R * self.eta ** (-self.s))

        if self.inner_loop_cnt == 0:
            print("Suggest a new batch of configurations for the new inner loop.")
            # Suggest a new batch of configurations.
            start_time = time.time()
            self.T = []
            for _ in range(n):
                self.T.append(self.mf_advisor.get_suggestion())
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
        self.tmp_history_dict = history_dict

        # Ensure that the history contains all the returned configurations
        assert (len(self.tmp_history_dict) == len(self.T))
        self.T = list(self.tmp_history_dict.keys())
        self.val_losses = list(self.tmp_history_dict.values())

        for config in history_dict:
            self.target_x[int(self.n_resource)].append(config)
            self.target_y[int(self.n_resource)].append(history_dict[config])

        if self.n_resource == self.R:
            self.incumbent_configs.extend(self.T)
            self.incumbent_perfs.extend(self.val_losses)

            # Update the underlying advisor
            for config in history_dict:
                observation = Observation(config=config, objectives=[history_dict[config]])
                self.mf_advisor.update_observation(observation)
            print("BOHB optimizer updated!")

            inc_idx = np.argmin(np.array(self.incumbent_perfs))
            self.perfs = self.incumbent_perfs
            self.configs = self.incumbent_configs
            self.incumbent_perf = -self.incumbent_perfs[inc_idx]
            self.incumbent_config = self.incumbent_configs[inc_idx]
