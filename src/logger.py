
# Import the logging module to record application events
import logging

# Import os to work with file and directory paths
import os

# Import datetime to generate a unique log file name using date and time
from datetime import datetime


# Create a log file name using the current date and time
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Create the full path of the logs folder
logs_path = os.path.join(os.getcwd(), "logs")

# Create the logs folder if it does not already exist
os.makedirs(logs_path, exist_ok=True)

# Create the full path of the log file
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure the logging system
logging.basicConfig(
    filename=LOG_FILE_PATH,

    # Define the format of each log message
    # Includes date, time, line number, log level, and message
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
    
