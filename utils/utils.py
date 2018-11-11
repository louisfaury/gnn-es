import logging

CH = logging.StreamHandler()
CH.setLevel(logging.WARNING)
CH.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))


def get_or_default(config, key, default, logger):
    if key in config:
        return config[key]
    else:
        if 'copy' not in logger.name:
            logger.addHandler(CH)
            logger.warning("using default parameter (" + str(default) + ") for key (" + key + ")")
        return default


def log(message, logger):
    logger.addHandler(CH)
    logger.warning(message)
