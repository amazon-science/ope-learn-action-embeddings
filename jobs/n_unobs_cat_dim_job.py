from jobs.abstracts.abstract_synthetic_job import AbstractSyntheticJob


class NUnobsCatDimJob(AbstractSyntheticJob):
    def main(self, cfg):
        return self.run(cfg, "n_unobserved_cat_dim", cfg.n_unobserved_cat_dim_list)
