from jobs.abstracts.abstract_synthetic_job import AbstractSyntheticJob


class NValDataJob(AbstractSyntheticJob):
    def main(self, cfg):
        return self.run(cfg, "n_val_data", cfg.n_val_data_list)
