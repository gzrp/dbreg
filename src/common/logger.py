import os
import logging
from logging.handlers import TimedRotatingFileHandler


def get_logger(name, folder_name):
    if not os.path.exists(f"./{folder_name}"):
        os.makedirs(f"./{folder_name}")
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    filename = f"./{folder_name}/{name}.log"
    fh = TimedRotatingFileHandler(filename, when='D', backupCount=7)
    sh = logging.StreamHandler()

    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

if __name__ == '__main__':
    logger1 = get_logger("logger1", "log1")
    logger1.info("logger1.info")
    logger1.debug("logger1.debug")
    logger1.warning("logger1.warning")
    logger1.error("logger1.error")
    logger1.critical("logger1.critical")

    logger2 = get_logger("logger2", "log2")
    logger2.info("logger2.info")
    logger2.debug("logger2.debug")
    logger2.warning("logger2.warning")
    logger2.error("logger2.error")
    logger2.critical("logger2.critical")


