## Learning Action Embeddings for Off-Policy Evaluation 

This is the code repository that evaluates the proposed methods in the paper.

To get started, we recommend to check the [Example.ipynb](Example.ipynb) notebook as it clearly demonstrates benefits of the proposed method from Section 3 and implements everything in a few lines of code. To run the notebook, you only need `python 3` with standard machine learning libraries.

To run the other synthetic and real world experiments in the paper, you might need the AWS account as everything is implemented to run with AWS SageMaker. Depending on training instance used, the experiments may run for a couple of hours/days.

### Setting up the environment
The following are the steps to setup the environment

1. Create an AWS account with the access to *S3 Bucket* and *SageMaker*
1. [Configure](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html) your AWS account locally
1. In the [constants.py](experiments/utils/constants.py) file, specify the S3 Bucket name and IAM execution role
1. run `virtualenv -p python3 ope_enviroment`
1. run `source ./ope_enviroment/bin/activate`
1. run `pip install -r requirements.txt`

### Running the Experiments
To reproduce the synthetic experiments, it is important to set the right parameters for the dataset generation in [configs.py](experiments/utils/configs.py?plain=1#L31) before running the commands.
The file contains default parameters and we prompt you to change it for the specific experiment if necessary. Note that the numbering is shifted to include only experiment plots.

#### Figure 1 - Can we improve over standard baselines without using pre-defined embeddings?
Varying the number of actions:\
`python cli.py run -c "NActionsExperiment" -n "Figure1A"`

Varying the number of samples in the logged data:\
`python cli.py run -c "NValDataExperiment" -n "Figure1B"`

#### Figure 2 - Can we improve upon high-variance pre-defined embeddings?
[configs.py](experiments/utils/configs.py?plain=1#L31), `n_cat_dim=20`\
Varying the number of actions:\
`python cli.py run -c "NActionsExperiment" -n "Figure2A"`

Varying the number of samples in the logged data:\
`python cli.py run -c "NValDataExperiment" -n "Figure2B"`

#### Figure 3 - High-bias, high-variance pre-defined embeddings
Varying the number of unobserved dimensions:\
`python cli.py run -c "NUnobsCatDimExperiment" -n "Figure3"`

#### Figure 4 - High-bias, low variance pre-defined embeddings
[configs.py](experiments/utils/configs.py?plain=1#L31), `n_cat_dim=4`, `n_unobserved_cat_dim=2`\
Varying the number of actions:\
`python cli.py run -c "NActionsExperiment" -n "Figure4A"`

Varying the number of samples in the logged data:\
`python cli.py run -c "NValDataExperiment" -n "Figure4B"`

#### Figure 5 and 8 - Real-world data, including the appendix
Varying the number of samples in the logged data:\
`python cli.py run -c "RealDatasetExperiment" -n "Figure5"`

#### Figure 6 - Appendix: Can we improve when the reward model for learning embeddings does not match the reward function?
[abstract_synthetic_job.py](jobs/abstracts/abstract_synthetic_job.py?plain=1#L50), comment out line 49 and uncomment line 50\
Varying the number of actions:\
`python cli.py run -c "NActionsExperiment" -n "Figure6A"`

Varying the number of samples in the logged data:\
`python cli.py run -c "NValDataExperiment" -n "Figure6B"`

#### Figure 7 - Appendix: How does the size of the learned embedding impact bias and variance of the estimate?
[configs.py](experiments/utils/configs.py?plain=1#L33), `dim_context=100`\
Varying the number of actions:\
`python cli.py run -c "AblationHpoExperiment" -n "Figure7"`

### Producing the figures
Once the training jobs have finished, you can produce the figures by calling the `output` command of `cli.py` with the same arguments.
For example, to create the output for Figure 1, the command is:

`python cli.py output -c "NActionsExperiment" -n "Figure1A"`

The output is then produced in the [log](log) directory.
To omit some of the methods from displaying, you need to modify the corresponding experiment class file, for example, comment out [line 64](experiments/n_actions_experiment.py?plain=1#L64) in the `exclude` list to hide the *MIPS (true)* estimator in the *varying the number of actions* experiments.\
To adjust the legend so it fits single-column figures, edit the number of columns `n_col` in the [plots.py](experiments/utils/plots.py?plain=1#L43) (for the most cases `n_col=4`).

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

