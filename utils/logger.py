import logging

LOG_FORMAT = '%(asctime)s-%(levelname)s-%(filename)s[%(lineno)d]-%(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)


def create_logger(log_fp, name=None):

    handler = logging.FileHandler(log_fp)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.addHandler(handler)
    return logger

