
# Import os module to create folders and manage file paths
import os

# Import sys module for exception details
import sys

# Import CustomException to handle errors with details
from src.exception import CustomException

# Import logging to record program execution and errors
from src.logger import logging

# Import pandas to read and manipulate datasets
import pandas as pd

# Import function to split dataset into training and testing sets
from sklearn.model_selection import train_test_split

# Import dataclass for configuration classes
from dataclasses import dataclass

# Import data transformation classes
from src.components.data_transformation import (
    DataTransformation,
    DataTransformationConfig,
)

# Import model training classes
from src.components.model_trainer import (
    ModelTrainerConfig,
    ModelTrainer,
)


# Store paths for the output files
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


# Handle the data ingestion process
class DataIngestion:

    def __init__(self):
        # Initialize the output file paths
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")

        try:
            # Read the source dataset
            df = pd.read_csv("stud.csv")
            logging.info("Dataset loaded successfully")

            # Create the artifacts folder if it does not exist
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path),
                exist_ok=True,
            )

            # Save the original dataset
            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True,
            )

            logging.info("Starting train-test split")

            # Split data: 80% training and 20% testing
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
            )

            # Save training data
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True,
            )

            # Save testing data
            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True,
            )

            logging.info("Data ingestion completed successfully")

            # Return the paths of the training and testing datasets
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            logging.exception("Error occurred during data ingestion")
            raise CustomException(e, sys) from e


# Execute the pipeline when this file is run directly
if __name__ == "__main__":
    try:
        # Step 1: Data ingestion
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()

        # Step 2: Data transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = (
            data_transformation.initiate_data_transformation(
                train_data,
                test_data,
            )
        )

        # Step 3: Model training
        model_trainer = ModelTrainer()
        model_score = model_trainer.initiate_model_trainer(
            train_arr,
            test_arr,
        )

        print("Model training completed.")
        print("Model score:", model_score)

    except Exception as e:
        logging.exception("Pipeline execution failed")
        raise
