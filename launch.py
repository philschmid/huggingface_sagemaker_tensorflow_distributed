import sagemaker
import os
import boto3
from sagemaker.huggingface import HuggingFace

os.environ["AWS_PROFILE"] = "hf-sm"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
iam_client = boto3.client("iam")
role = iam_client.get_role(RoleName="sagemaker_execution_role")["Role"]["Arn"]
sess = sagemaker.Session()

# hyperparameters, which are passed into the training job
hyperparameters = {
    "epochs": 1,
    "train_batch_size": 8,
    "eval_batch_size": 2,
    "model_name_or_path": "bert-large-uncased-whole-word-masking",
}
# configuration for running training on smdistributed Data Parallel
# distribution = {'smdistributed':{'dataparallel':{ 'enabled': True }}}
# horovod launch
# distribution = {"mpi": {"enabled": True, "custom_mpi_options": "-verbose --NCCL_DEBUG=INFO -x RDMAV_FORK_SAFE=1"}}
# no distribution
distribution = None
# instance configurations
# instance_type = "ml.p3.16xlarge"
instance_type = "ml.p3dn.24xlarge"
instance_count = 1
# volume_size=200

huggingface_estimator = HuggingFace(
    # distibuted script,
    # entry_point="train.py",
    # single_node script,
    entry_point="singe_node_train.py",
    source_dir="./scripts",
    instance_type=instance_type,
    role=role,
    session=sess,
    instance_count=instance_count,
    transformers_version="4.5.0",
    tensorflow_version="2.4.1",
    py_version="py37",
    distribution=distribution,
    hyperparameters=hyperparameters,
    base_job_name="hf-tf-bert-" + str(instance_count) + "node-" + instance_type.replace(".", "-"),
    debugger_hook_config=False,  # currently needed
)
huggingface_estimator.fit()
