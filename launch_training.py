from sagemaker.train import ModelTrainer
from sagemaker.train.configs import SourceCode, InputData, Compute, OutputDataConfig
from sagemaker.core.s3 import S3Uploader
from sagemaker.core import image_uris

bucket = "test"
prefix = "qwen3"

# 上传数据和代码到 S3
S3Uploader.upload(local_path="data", desired_s3_uri=f"s3://{bucket}/{prefix}/data")
S3Uploader.upload(local_path="code", desired_s3_uri=f"s3://{bucket}/{prefix}/code")

# 配置源代码
source_code = SourceCode(
    source_dir="./code",
    entry_script="train_qwen3_lora.py",
)

# 配置训练数据输入
train_data = InputData(
    channel_name="train",
    data_source=f"s3://{bucket}/{prefix}/data",
)

# 配置计算资源
compute = Compute(
    instance_type='ml.g5.xlarge',
    instance_count=1,
)

# 配置输出路径
output_data_config = OutputDataConfig(
    s3_output_path=f"s3://{bucket}/{prefix}/output/"
)

# pytorch_image = image_uris.retrieve(
#     framework='pytorch',
#     region='us-east-2',
#     version='2.7.1',
#     py_version='py312',
#     instance_type='ml.g5.xlarge',
#     image_scope='training'
# )
# print(pytorch_image)

# 创建 ModelTrainer (使用 PyTorch 官方镜像)
trainer = ModelTrainer(
    training_image="763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training:2.7.1-gpu-py312",
    source_code=source_code,
    role="arn:aws:iam::543687745601:role/service-role/AmazonSageMakerAdminIAMExecutionRole",
    compute=compute,
    output_data_config=output_data_config,
    hyperparameters={"model_id": "Qwen/Qwen3-1.7B"},
)

# 启动训练
trainer.train(input_data_config=[train_data])
