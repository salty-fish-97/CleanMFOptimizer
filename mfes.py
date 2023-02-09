import numpy as np

from openbox.core.mf_batch_advisor import MFBatchAdvisor
from openbox import Observation

from base import BaseMFOptimizer


class MFESOptimizer(BaseMFOptimizer):
    def __init__(self, config_space, seed=1, R=27, eta=3, n_jobs=1):
        super().__init__(config_space=config_space,
                         seed=seed, R=R,
                         eta=eta, n_jobs=n_jobs)

        self.name = 'MFES'
        self.mf_advisor = MFBatchAdvisor(config_space=config_space,
                                         surrogate_type='mfgpe')

    def update_observations(self, history_dict):
        self.tmp_history_dict = history_dict

        # Ensure that the history contains all the returned configurations
        assert (len(self.tmp_history_dict) == len(self.T))
        self.T = list(self.tmp_history_dict.keys())
        self.val_losses = list(self.tmp_history_dict.values())

        for config in history_dict:
            self.target_x[int(self.n_resource)].append(config)
            self.target_y[int(self.n_resource)].append(history_dict[config])

        for config in history_dict:
            observation = Observation(config=config, objectives=[history_dict[config]])
            self.mf_advisor.update_observation(observation, self.n_resource / self.R)
        print("MFES optimizer updated!")

        if self.n_resource == self.R:
            self.incumbent_configs.extend(self.T)
            self.incumbent_perfs.extend(self.val_losses)

            inc_idx = np.argmin(np.array(self.incumbent_perfs))
            self.perfs = self.incumbent_perfs
            self.configs = self.incumbent_configs
            self.incumbent_perf = -self.incumbent_perfs[inc_idx]
            self.incumbent_config = self.incumbent_configs[inc_idx]
