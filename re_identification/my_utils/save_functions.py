import os
import pickle


def save_state(evaluator, path):
    with open(path, 'wb') as f:
        pickle.dump(evaluator, f)


def load_state(evaluator, logger, path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            loaded_evaluator = pickle.load(f)
        evaluator.__dict__.update(loaded_evaluator.__dict__)
        logger.info("Loaded evaluator state from {}".format(path))
    else:
        logger.info("No previous evaluator state found, creating new one")
        evaluator.reset()


def save_batch_index(idx, path):
    with open(path, 'wb') as f:
        pickle.dump(idx, f)


def load_batch_index(logger, path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            batch_index = pickle.load(f)
            logger.info("Resuming from batch index {}".format(batch_index))
            return batch_index
    return 0
