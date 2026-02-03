from angkorice_vision.logging.logger import logging
from angkorice_vision.exception.exception import AngkoriceVisionException
import sys

def sum():
    try:
        pass
        logging.info("Entering the sumation funciton!")

        a = input("Enter first number :") # input return a string
        b = input("Enter second number :")
        c = int(a) + int(b)
        logging.info("Exit the sumation !")
        return c
    
    except Exception as e:
        raise AngkoriceVisionException(e, sys)

result = sum()
print(result)

if __name__=="__main__":
    sum()