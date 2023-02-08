import time
import numpy as np
from math import log, ceil

from openbox import Observation
from openbox.core.sync_batch_advisor import SyncBatchAdvisor

from base import BaseMFOptimizer


class BOHBOptimizer(BaseMFOptimizer):
    def __init__(self, config_space, mode='smac', seed=1, R=27, eta=3, n_jobs=1):
        super().__init__(config_space=config_space,
                         seed=seed, R=R,
                         eta=eta, n_jobs=n_jobs)

        self.mode = mode
        self.mf_advisor = SyncBatchAdvisor(config_space=config_space,
                                           surrogate_type='prf',
                                           acq_type='ei',
                                           acq_optimizer_type='local_random')

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
                observation = Observation(config=config, objs=[history_dict[config]])
                self.mf_advisor.update_observation(observation)
            print("BOHB optimizer updated!")

            inc_idx = np.argmin(np.array(self.incumbent_perfs))
            self.perfs = self.incumbent_perfs
            self.configs = self.incumbent_configs
            self.incumbent_perf = -self.incumbent_perfs[inc_idx]
            self.incumbent_config = self.incumbent_configs[inc_idx]
