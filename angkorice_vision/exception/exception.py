import sys
import os
from angkorice_vision.logging.logger import logging

class AngkoriceVisionException(Exception):
    def __init__(self, error_message: Exception, error_details: sys):
        super().__init__(error_message)

        self.error_message = error_message
        _,_, exc_tb = error_details.exc_info()
        self.lineno = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    
    def __str__(self):
        return "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
            self.file_name, self.lineno, str(self.error_message)
        )