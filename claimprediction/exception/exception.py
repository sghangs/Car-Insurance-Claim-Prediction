import sys
from claimprediction.logging import logger

class CarInsuranceException(Exception):
    def __init__(self,error_message):
        self.error_message=error_message
        _,_,exc_tb=sys.exc_info()

        self.line_no=exc_tb.tb_lineno
        self.file_name=exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return "Error occured in python script name {0} line number {1} error message {2}".format(
            self.file_name,self.line_no,str(self.error_message)
        )

if __name__=="__main__":
    try:
        logger.logging.info("Entry the try block")
        a=1/0
        print("This is not going to be printed")
    except Exception as e:
        raise CarInsuranceException(e)

