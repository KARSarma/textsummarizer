import yaml
from pathlib import Path
from src.textSummarizer.constants import CONFIG_FILE_PATH
from src.textSummarizer.entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)


class ConfigurationManager:
    def __init__(self, config_file_path: str = CONFIG_FILE_PATH):
        self.config = self._load_config(config_file_path)

    def _load_config(self, config_file_path: str):
        try:
            with open(config_file_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            raise Exception(f"Failed to load config file: {config_file_path}. Error: {e}")

    def get_data_ingestion_config(self):
        data_ingestion_config = self.config["data_ingestion"]
        return DataIngestionConfig(
            root_dir=Path(data_ingestion_config["root_dir"]),
            source_url=data_ingestion_config["source_URL"],
            local_data_file=Path(data_ingestion_config["local_data_file"]),
            unzip_dir=Path(data_ingestion_config["unzip_dir"])
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        data_transformation_config = self.config["data_transformation"]
        return DataTransformationConfig(
            root_dir=Path(data_transformation_config["root_dir"]),
            data_path=Path(data_transformation_config["data_path"]),
            tokenizer_name=data_transformation_config["tokenizer_name"]
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        model_trainer_config = self.config["model_trainer"]
        return ModelTrainerConfig(
            root_dir=Path(model_trainer_config["root_dir"]),
            data_path=Path(model_trainer_config["data_path"]),
            model_ckpt=model_trainer_config["model_ckpt"],
            num_train_epochs=model_trainer_config.get("num_train_epochs", 1),
            warmup_steps=model_trainer_config.get("warmup_steps", 500),
            per_device_train_batch_size=model_trainer_config.get("per_device_train_batch_size", 1),
            per_device_eval_batch_size=model_trainer_config.get("per_device_eval_batch_size", 1),
            weight_decay=model_trainer_config.get("weight_decay", 0.01),
            logging_steps=model_trainer_config.get("logging_steps", 10),
            evaluation_strategy=model_trainer_config.get("evaluation_strategy", "steps"),
            eval_steps=model_trainer_config.get("eval_steps", 500),
            save_steps=model_trainer_config.get("save_steps", 1000000),
            gradient_accumulation_steps=model_trainer_config.get("gradient_accumulation_steps", 16),
            train_subset=model_trainer_config.get("train_subset", 200),
            eval_subset=model_trainer_config.get("eval_subset", 50)
        )
