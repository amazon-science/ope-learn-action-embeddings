# Learning Action Embeddings for Off-Policy Evaluation 

This repository contains code for evaluating the methods proposed in [Learning Action Embeddings for Off-Policy Evaluation](https://arxiv.org/abs/2305.03954).

To get started, we recommend checking the [Example.ipynb](Example.ipynb) notebook as it clearly demonstrates benefits of the proposed method from Section 3 and implements everything in a few lines of code. To run the notebook, you only need `python 3` with standard machine learning libraries.

To run the other synthetic and real-world experiments in the paper, you might need the AWS account as everything is implemented to run with AWS SageMaker. Depending on the training instance used, the experiments may run for a couple of hours/days.

We also provide commands to run the experiments locally (requires considerable computational resource).

## Setting up the environment
The following are the steps to set up the environment. You can skip the first three steps if you want to run experiments locally.

1. Create an AWS account with access to *S3 Bucket* and *SageMaker* (follow [SageMaker getting started guide](https://docs.aws.amazon.com/sagemaker/latest/dg/gs.html) and [this Jupyter notebook](https://docs.aws.amazon.com/sagemaker/latest/dg/ex1-train-model.html) to test if the access works correctly)
1. [Configure](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html) your AWS account locally
1. In the [constants.py](experiments/utils/constants.py) file, specify the S3 Bucket name and IAM execution role
1. run `virtualenv -p python3 .venv`
1. run `source ./.venv/bin/activate`
1. run `pip install -r requirements.txt`

## Running the Experiments
To reproduce the synthetic experiments on SageMaker, it is important to set the right parameters for the dataset generation in [configs.py](experiments/utils/configs.py?plain=1#L31) before running the commands (this is not needed when running experiments locally).
The file contains default parameters and we prompt you to change it for the specific experiment if necessary. Note that the numbering is shifted to include only experiment plots.

### Figure 1 - Can we improve over standard baselines without using pre-defined embeddings?
Varying the number of actions:\
`python cli.py run -c "NActionsExperiment" -n "Figure1A"`\
To execute locally: `python entry_point.py --config '{"job_class_name": "NActionsJob", "s3_path": "results/Figure1A"}'`

Varying the number of samples in the logged data:\
`python cli.py run -c "NValDataExperiment" -n "Figure1B"`\
To execute locally: `python entry_point.py --config '{"job_class_name": "NValDataJob", "s3_path": "results/Figure1B"}'`

### Figure 2 - Can we improve upon high-variance pre-defined embeddings?
[configs.py](experiments/utils/configs.py?plain=1#L31), `n_cat_dim=20`\
Varying the number of actions:\
`python cli.py run -c "NActionsExperiment" -n "Figure2A"`\
To execute locally: `python entry_point.py --config '{"job_class_name": "NActionsJob", "s3_path": "results/Figure2A", "n_cat_dim": 20}'`

Varying the number of samples in the logged data:\
`python cli.py run -c "NValDataExperiment" -n "Figure2B"`\
To execute locally: `python entry_point.py --config '{"job_class_name": "NValDataJob", "s3_path": "results/Figure2B", "n_cat_dim": 20}'`

### Figure 3 - High-bias, high-variance pre-defined embeddings
Varying the number of unobserved dimensions:\
`python cli.py run -c "NUnobsCatDimExperiment" -n "Figure3"`\
To execute locally: `python entry_point.py --config '{"job_class_name": "NUnobsCatDimJob", "s3_path": "results/Figure3"}'`

### Figure 4 - High-bias, low-variance pre-defined embeddings
[configs.py](experiments/utils/configs.py?plain=1#L31), `n_cat_dim=4`, `n_unobserved_cat_dim=2`\
Varying the number of actions:\
`python cli.py run -c "NActionsExperiment" -n "Figure4A"`\
To execute locally: `python entry_point.py --config '{"job_class_name": "NActionsJob", "s3_path": "results/Figure4A", "n_cat_dim": 4, "n_unobserved_cat_dim": 2}'`

Varying the number of samples in the logged data:\
`python cli.py run -c "NValDataExperiment" -n "Figure4B"`\
To execute locally: `python entry_point.py --config '{"job_class_name": "NValDataJob", "s3_path": "results/Figure4B", "n_cat_dim": 4, "n_unobserved_cat_dim": 2}'`

### Figure 5 and 8 - Real-world data, including the appendix
You need to download [the data](https://research.zozo.com/data_release/open_bandit_dataset.zip) and upload the extracted folder `open_bandit_dataset` to your S3 bucket (same as in [constants.py](experiments/utils/constants.py))):\
`python cli.py run -c "RealDatasetExperiment" -n "Figure5"`\
To execute locally: `python entry_point.py --config '{"job_class_name": "RealDatasetJob", "s3_path": "results/Figure5"}'`

### Figure 6 - Appendix: Can we improve when the reward model for learning embeddings does not match the reward function?
[abstract_synthetic_job.py](jobs/abstracts/abstract_synthetic_job.py?plain=1#L50), comment out line 49 and uncomment line 50\
Varying the number of actions:\
`python cli.py run -c "NActionsExperiment" -n "Figure6A"`\
To execute locally (need to change the source code as well): `python entry_point.py --config '{"job_class_name": "NActionsJob", "s3_path": "results/Figure6A"}'`

Varying the number of samples in the logged data:\
`python cli.py run -c "NValDataExperiment" -n "Figure6B"`\
To execute locally (need to change the source code as well): `python entry_point.py --config '{"job_class_name": "NValDataJob", "s3_path": "results/Figure6B"}'`

### Figure 7 - Appendix: How does the size of the learned embedding impact bias and variance of the estimate?
[configs.py](experiments/utils/configs.py?plain=1#L33), `dim_context=100`\
Varying the number of actions:\
`python cli.py run -c "AblationHpoExperiment" -n "Figure7"`\
This experiment cannot be run locally as it is using SageMaker hyperparameter optimization.

## Producing the figures
Once the SageMaker training jobs have finished, you can produce the figures by calling the `output` command of `cli.py` with the same arguments.
For example, to create the output for Figure 1, the command is:

`python cli.py output -c "NActionsExperiment" -n "Figure1A"`

If you have the results stored locally, you can provide the directory where they are stored instead:

`python cli.py output -c "NActionsExperiment" -d "results/Figure1A"`

The output is then produced in the [results](results) directory.
To omit some of the methods from displaying, you need to modify the corresponding experiment class file, for example, comment out [line 64](experiments/n_actions_experiment.py?plain=1#L64) in the `exclude` list to hide the *MIPS (true)* estimator in the *varying the number of actions* experiments.\
To adjust the legend so it fits single-column figures, edit the number of columns `n_col` in the [plots.py](experiments/utils/plots.py?plain=1#L43) (for the most cases `n_col=4`).

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

## Citation
```
@misc{cief2023,
  title = {Learning Action Embeddings for Off-Policy Evaluation},
  author = {Cief, Matej and Golebiowski, Jacek and Schmidt, Philipp and Abedjan, Ziawasch and Bekasov, Artur},
  year = {2023},
  month = may,
  number = {arXiv:2305.03954},
  eprint = {2305.03954},
  primaryclass = {cs},
  publisher = {{arXiv}},
  doi = {10.48550/arXiv.2305.03954},
}
```
