
# Object ko file mein save karne ke liye pickle import karte hain
import os
import sys
import pickle

# Custom error handling ke liye
from src.exception import CustomException

# Project logs record karne ke liye
from src.logger import logging


# Python object ko file mein save karne wala function
def save_object(file_path, obj):
    try:
        # File ke parent folder ka path nikalte hain
        dir_path = os.path.dirname(file_path)

        # Agar folder nahi hai, toh create karte hain
        os.makedirs(dir_path, exist_ok=True)

        # File ko binary write mode mein open karte hain
        with open(file_path, "wb") as file_obj:

            # Object ko pickle format mein save karte hain
            pickle.dump(obj, file_obj)

        # Successful saving ka log record karte hain
        logging.info(f"Object saved successfully at: {file_path}")

    # Error aaye toh usse handle karte hain
    except Exception as e:
        raise CustomException(e, sys)


# Saved pickle object ko load karne wala function
def load_object(file_path):
    try:
        # File ko binary read mode mein open karte hain
        with open(file_path, "rb") as file_obj:

            # Saved object ko load karke return karte hain
            return pickle.load(file_obj)

    # Error aaye toh custom exception raise karte hain
    except Exception as e:
        raise CustomException(e, sys)
