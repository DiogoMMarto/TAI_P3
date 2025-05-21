from config import LOG_LEVEL
import datetime

color = {
    "INFO": "\033[94m",
    "WARNING": "\033[93m", 
    "ERROR": "\033[91m",  
    "RESET": "\033[0m"  
}

def log(level, *args):
    """
    Logs a message with the specified logging level.
    """
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if level == "INFO" and LOG_LEVEL != "INFO":
        return
    if level == "WARNING" and LOG_LEVEL == "ERROR":
        return
    print(f"{color[level]}[{date}] [{level}] {' '.join(map(str, args))}{color['RESET']}")
        