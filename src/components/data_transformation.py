
# System-related functions aur exception handling ke liye sys import karte hain
import sys

# Configuration class ko simple banane ke liye dataclass import karte hain
from dataclasses import dataclass

# Numerical arrays aur data ko combine karne ke liye NumPy import karte hain
import numpy as np

# CSV data read karne ke liye Pandas import karte hain
import pandas as pd

# Numerical aur categorical columns par alag preprocessing apply karne ke liye
from sklearn.compose import ColumnTransformer

# Missing values fill karne ke liye SimpleImputer import karte hain
from sklearn.impute import SimpleImputer

# Multiple preprocessing steps ko sequence mein execute karne ke liye
from sklearn.pipeline import Pipeline

# Categorical encoding aur numerical scaling ke liye
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Project ki custom error-handling class import karte hain
from src.exception import CustomException

# Project mein execution logs record karne ke liye
from src.logger import logging

# File paths aur folders manage karne ke liye
import os

# Preprocessing object ko file mein save karne ke liye
from src.utils import save_object


# Data transformation ki configuration define karte hain
@dataclass
class DataTransformationConfig:

    # Preprocessor pickle file ka path define karte hain
    preprocessor_obj_file_path: str = os.path.join(
        "artifacts", "preprocessor.pkl"
    )


# Data transformation ka complete workflow is class mein hoga
class DataTransformation:

    # Class ka object create hone par constructor execute hota hai
    def __init__(self):

        # Configuration object create karte hain
        self.data_transformation_config = DataTransformationConfig()

    # Numerical aur categorical preprocessing pipelines create karte hain
    def get_data_transformer_object(self):

        # Errors ko handle karne ke liye try block
        try:

            # Numerical columns ki list
            numerical_columns = [
                "writing_score",
                "reading_score"
            ]

            # Categorical columns ki list
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Numerical data ke liye pipeline create karte hain
            num_pipeline = Pipeline(
                steps=[
                    # Missing numerical values ko median se fill karte hain
                    ("imputer", SimpleImputer(strategy="median")),

                    # Numerical features ko standardize karte hain
                    ("scaler", StandardScaler()),
                ]
            )

            # Categorical data ke liye pipeline create karte hain
            cat_pipeline = Pipeline(
                steps=[
                    # Missing categorical values ko most frequent value se fill karte hain
                    ("imputer", SimpleImputer(strategy="most_frequent")),

                    # Categories ko numerical indicator columns mein convert karte hain
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),

                    # Encoded features ko scale karte hain
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            # Numerical pipeline ready hone ka log record karte hain
            logging.info("Numerical preprocessing pipeline created")

            # Categorical pipeline ready hone ka log record karte hain
            logging.info("Categorical preprocessing pipeline created")

            # Dono pipelines ko ColumnTransformer mein combine karte hain
            # sparse_threshold=0 output ko dense array banane mein madad karta hai
            preprocessor = ColumnTransformer(
                transformers=[
                    # Numerical columns par numerical pipeline apply hoti hai
                    ("num_pipeline", num_pipeline, numerical_columns),

                    # Categorical columns par categorical pipeline apply hoti hai
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ],
                sparse_threshold=0,
            )

            # Complete preprocessing object return karte hain
            return preprocessor

        # Pipeline banate waqt error aaye toh yeh block execute hota hai
        except Exception as e:

            # Error ko custom exception ke through raise karte hain
            raise CustomException(e, sys)

    # Train aur test datasets ko transform karne ka method
    def initiate_data_transformation(self, train_path, test_path):

        # Data transformation workflow ko try block mein execute karte hain
        try:

            # Training CSV file ko DataFrame mein read karte hain
            train_df = pd.read_csv(train_path)

            # Testing CSV file ko DataFrame mein read karte hain
            test_df = pd.read_csv(test_path)

            # Successful data reading ka log record karte hain
            logging.info("Reading train and test data completed")

            # Preprocessing object create karne ka log record karte hain
            logging.info("Obtaining preprocessing object")

            # Numerical aur categorical pipelines wala object create karte hain
            preprocessing_obj = self.get_data_transformer_object()

            # Target column jise model predict karega
            target_column_name = "math_score"

            # Training data se input features alag karte hain
            input_feature_train_df = train_df.drop(
                columns=[target_column_name]
            )

            # Training data ka target alag karte hain
            target_feature_train_df = train_df[target_column_name]

            # Testing data se input features alag karte hain
            input_feature_test_df = test_df.drop(
                columns=[target_column_name]
            )

            # Testing data ka target alag karte hain
            target_feature_test_df = test_df[target_column_name]

            # Training data par preprocessing fit aur apply karte hain
            # fit: training data se imputation, encoding aur scaling parameters learn karta hai
            # transform: learned parameters ko data par apply karta hai
            logging.info("Applying preprocessing to training data")

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )

            # Testing data par training mein learned parameters apply karte hain
            # Testing data par dobara fit nahi karte
            logging.info("Applying preprocessing to testing data")

            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df
            )

            # Transformed training features aur target ko combine karte hain
            train_arr = np.c_[
                input_feature_train_arr,
                target_feature_train_df.to_numpy(),
            ]

            # Transformed testing features aur target ko combine karte hain
            test_arr = np.c_[
                input_feature_test_arr,
                target_feature_test_df.to_numpy(),
            ]

            # Preprocessor save karne se pehle artifacts folder create karte hain
            os.makedirs(
                os.path.dirname(
                    self.data_transformation_config.preprocessor_obj_file_path
                ),
                exist_ok=True,
            )

            # Preprocessor ko pickle file mein save karte hain
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            # Saving complete hone ka log record karte hain
            logging.info("Preprocessor object saved successfully")

            # Training array, testing array aur preprocessor path return karte hain
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        # Transformation ke dauran error aaye toh usse handle karte hain
        except Exception as e:

            # Detailed error ke saath custom exception raise karte hain
            raise CustomException(e, sys)
