from src.textSummarizer.logging import logger
from src.textSummarizer.pipeline.stage_1_data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.textSummarizer.pipeline.stage_2_data_transformation_pipeline import DataTransformationTrainingPipeline
from src.textSummarizer.pipeline.stage_3_model_trainer_pipeline import ModelTrainerTrainingPipeline
from src.textSummarizer.pipeline.stage_4_model_evaluation import ModelEvaluationTrainingPipeline

def run_pipeline():
    stages = [
        {
            "name": "Data Ingestion stage",
            "pipeline": DataIngestionTrainingPipeline,
            "method": "initiate_data_ingestion",
        },
        {
            "name": "Data Transformation stage",
            "pipeline": DataTransformationTrainingPipeline,
            "method": "initiate_data_transformation",
        },
        {
            "name": "Model Trainer stage",
            "pipeline": ModelTrainerTrainingPipeline,
            "method": "initiate_model_trainer",
        },
        {
            "name": "Model Evaluation stage",
            "pipeline": ModelEvaluationTrainingPipeline,
            "method": "initiate_model_evaluation",
        },
    ]

    for stage in stages:
        try:
            logger.info(f"stage {stage['name']} initiated")
            pipeline_instance = stage["pipeline"]()
            getattr(pipeline_instance, stage["method"])()
            logger.info(f"Stage {stage['name']} Completed")
        except Exception as e:
            logger.exception(e)
            raise e

if __name__ == "__main__":
    run_pipeline()
