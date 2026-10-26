
# os module ko import kar rahe hain
# Iska use folders aur file paths create karne ke liye hota hai
import os

# sys module ko import kar rahe hain
# Iska use system-related information aur exception details ke liye hota hai
import sys

# CustomException class ko import kar rahe hain
# Ye errors ko detailed format mein handle karne ke liye use hoti hai
from src.exception import CustomException

# Logging ko import kar rahe hain
# Isse program ke execution steps aur errors record kar sakte hain
from src.logger import logging

# Pandas library ko import kar rahe hain
# Iska use dataset ko read aur manipulate karne ke liye hota hai
import pandas as pd

# train_test_split function ko import kar rahe hain
# Iska use dataset ko training aur testing data mein divide karne ke liye hota hai
from sklearn.model_selection import train_test_split

# dataclass decorator ko import kar rahe hain
# Isse configuration store karne wali class banana easy hota hai
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

# @dataclass automatically constructor jaise methods generate karta hai
# Is class mein hum data ingestion ke output file paths define karenge
@dataclass
class DataIngestionConfig:

    # Training dataset ko save karne ka path
    train_data_path: str = os.path.join('artifacts', 'train.csv')

    # Testing dataset ko save karne ka path
    test_data_path: str = os.path.join('artifacts', 'test.csv')

    # Original dataset ko save karne ka path
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


# Data ingestion process ko manage karne ke liye class bana rahe hain
class DataIngestion:

    # Constructor method object create hone par automatically execute hota hai
    def __init__(self):

        # DataIngestionConfig class ka object create kar rahe hain
        # Is object mein saare output file paths available honge
        self.ingestion_config = DataIngestionConfig()

    # Ye method data load, split aur save karne ka kaam karta hai
    def initiate_data_ingestion(self):

        # Log message record karte hain ki data ingestion process start ho raha hai
        logging.info("Entered the data ingestion method or component")

        # try block mein woh code likhte hain jisme error aa sakta hai
        try:

            # stud.csv file ko read karke Pandas DataFrame mein load karte hain
            df = pd.read_csv('stud.csv')

            # Log karte hain ki dataset successfully read ho gaya
            logging.info('Read the dataset as dataframe')

            # Artifacts folder create karte hain agar woh pehle se exist nahi karta
            # exist_ok=True hone se existing folder par error nahi aata
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path),
                exist_ok=True
            )

            # Original dataset ko artifacts/data.csv mein save karte hain
            # index=False extra index column ko save hone se rokta hai
            # header=True column names ko CSV mein save karta hai
            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )

            # Log karte hain ki train-test split process start ho raha hai
            logging.info("Train test split initiated")

            # Dataset ko training aur testing sets mein divide karte hain
            # test_size=0.2 ka matlab 20% testing aur 80% training data
            # random_state=42 se split reproducible rehta hai
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            # Training dataset ko artifacts/train.csv mein save karte hain
            # index=False extra index column ko remove karta hai
            # header=True column names ko preserve karta hai
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )

            # Testing dataset ko artifacts/test.csv mein save karte hain
            # Is file ka use baad mein model evaluation ke liye hoga
            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )

            # Log karte hain ki data ingestion successfully complete ho gaya
            logging.info("Ingestion of the data is completed")

            # Training aur testing files ke paths return karte hain
            # Next component in paths ka use kar sakta hai
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        # Agar try block mein koi error aata hai, to except block execute hota hai
        except Exception as e:

            # Error ko CustomException ke through detailed format mein raise karte hain
            # sys se current exception ki details pass ki jaati hain
            raise CustomException(e, sys)


# Check karte hain ki file directly execute hui hai ya import ki gayi hai
# Direct execution par hi neeche ka code run hoga
if __name__ == "__main__":

    # DataIngestion class ka object create karte hain
    obj = DataIngestion()

    # Data ingestion method call karte hain
    # Ye dataset read, split aur save karega
    train_data,test_data=obj.initiate_data_ingestion()
    data_transformation=DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)
