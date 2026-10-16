
import sys


def error_message_detail(error, error_detail: sys):
    # Get the complete exception information
    _, _, exc_tb = error_detail.exc_info()

    # Get the Python file name where the error occurred
    file_name = exc_tb.tb_frame.f_code.co_filename

    # Create a detailed error message
    error_message = (
        "Error occurred in Python script name "
        "[{0}] line number [{1}] error message [{2}]"
        .format(
            file_name,
            exc_tb.tb_lineno,
            str(error)
        )
    )

    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        # Initialize the parent Exception class
        super().__init__(error_message)

        # Store the detailed error message
        self.error_message = error_message_detail(
            error_message, error_detail
        )

    def __str__(self):
        # Return the detailed error message
        return self.error_message
