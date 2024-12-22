from src.textSummarizer.config.configuration import ConfigurationManager
from src.textSummarizer.components.model_trainer import ModelTrainer
from src.textSummarizer.logging import logger

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def initiate_model_trainer(self):
        try:
            logger.info("Initializing Model Trainer...")
            
            # Initialize ConfigurationManager
            config = ConfigurationManager()

            # Retrieve model trainer configuration
            model_trainer_config = config.get_model_trainer_config()
            
            # Initialize and train the model
            model_trainer = ModelTrainer(config=model_trainer_config)
            model_trainer.train()

            logger.info("Model Trainer stage completed successfully.")
        except Exception as e:
            logger.exception("Error in Model Trainer Pipeline")
            raise e
