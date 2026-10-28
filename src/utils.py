
# Object ko file mein save karne ke liye pickle import karte hain
import os
import sys
import pickle

# Custom error handling ke liye
from src.exception import CustomException

# Project logs record karne ke liye
from src.logger import logging

# R2 score calculate karne ke liye
from sklearn.metrics import r2_score


# Python object ko file mein save karne wala function
def save_object(file_path, obj):
    try:
        # File ke parent folder ka path nikalte hain
        dir_path = os.path.dirname(file_path)

        # Agar parent folder diya gaya hai aur exist nahi karta,
        # toh folder create karte hain
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        # File ko binary write mode mein open karte hain
        with open(file_path, "wb") as file_obj:
            # Object ko pickle format mein save karte hain
            pickle.dump(obj, file_obj)

        # Successful saving ka log record karte hain
        logging.info(f"Object saved successfully at: {file_path}")

    except Exception as e:
        # Error ko custom exception ke through raise karte hain
        raise CustomException(e, sys) from e


# Saved pickle object ko load karne wala function
def load_object(file_path):
    try:
        # File ko binary read mode mein open karte hain
        with open(file_path, "rb") as file_obj:
            # Saved object ko load karke return karte hain
            return pickle.load(file_obj)

    except Exception as e:
        # Error ko custom exception ke through raise karte hain
        raise CustomException(e, sys) from e


# Multiple regression models ko train aur evaluate karne wala function
def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        # Har model ka test R2 score store karne ke liye dictionary
        report = {}

        # Dictionary ke har model par loop chalate hain
        for model_name, model in models.items():

            # Model ko training data par train karte hain
            model.fit(X_train, y_train)

            # Training data par prediction
            y_train_pred = model.predict(X_train)

            # Testing data par prediction
            y_test_pred = model.predict(X_test)

            # Training data ka R2 score
            train_model_score = r2_score(
                y_train, y_train_pred
            )

            # Testing data ka R2 score
            test_model_score = r2_score(
                y_test, y_test_pred
            )

            # Model ke scores log karte hain
            logging.info(
                f"{model_name} - "
                f"Train R2: {train_model_score:.4f}, "
                f"Test R2: {test_model_score:.4f}"
            )

            # Test R2 score ko report mein store karte hain
            report[model_name] = test_model_score

        # Sabhi models ke scores return karte hain
        return report

    except Exception as e:
        # Error ko custom exception ke through raise karte hain
        raise CustomException(e, sys) from e
