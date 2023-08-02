import logging
import test_use_logger


def test():
    logger = logging.getLogger("test")
    logger.debug("test")


if __name__ == "__main__":
    # init logging in main module
    log_format = f'%(asctime)s %(name)s "%(pathname)s", line %(lineno)d, %(levelname)s: %(message)s'
    logging.basicConfig(
        level=logging.DEBUG, format=log_format, datefmt="%Y-%m-%d %H:%M:%S"
    )

    # get logger
    logger = logging.getLogger("main")
    logger.debug("test")

    # use logger with the same loggingconfig in other module
    test_use_logger.print()
