from jobs.abstracts.abstract_synthetic_job import AbstractSyntheticJob


class NActionsJob(AbstractSyntheticJob):
    def main(self, cfg):
        return self.run(cfg, "n_actions", cfg.n_actions_list)
