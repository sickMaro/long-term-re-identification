import datetime
import logging
import time
from contextlib import contextmanager
import os
import pickle

import torch

from fastreid.utils.logger import log_every_n_seconds


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.
    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.
    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def preprocess_inputs(self, inputs):
        pass

    def process(self, inputs, outputs):
        """
        Process an input/output pair.
        Args:
            inputs: the inputs that's used to call the model.
            outputs: the return value of `model(input)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.
        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:
                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


def save_state(evaluator, path):
    state_to_save = {k: v for k, v in evaluator.__dict__.items() if k != 'cfg'}
    with open(path, 'wb') as f:
        pickle.dump(state_to_save, f)


def load_state(evaluator, logger, path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            loaded_state = pickle.load(f)
            for k, v in loaded_state.items():
                setattr(evaluator, k, v)
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


def inference_on_dataset(model, data_loader, evaluator):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    The model will be used in eval mode.
    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use
            :class:`DatasetEvaluators([])` if you only want to benchmark, but
            don't want to do any evaluation.
    Returns:
        The return value of `evaluator.evaluate()`
    """

    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader.dataset)))
    dataset_name = evaluator.cfg.DATASETS.KWARGS.split(':')[1]

    total = len(data_loader)  # inference data loader must have a fixed length

    evaluator_path = '../../evaluation_log/evaluation_state_{}.pkl'.format(
        evaluator.cfg.OUTPUT_DIR.split('/')[-1], dataset_name)
    batch_index_path = '../../evaluation_log/batch_index_{}.pkl'.format(
        evaluator.cfg.OUTPUT_DIR.split('/')[-1], dataset_name)

    load_state(evaluator, logger, evaluator_path)
    start_idx = load_batch_index(logger, batch_index_path)

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0

    model.eval()

    if start_idx < total:
        for idx, inputs in enumerate(data_loader):
            if idx <= start_idx:
                continue  # Skip the batches we've already processed
            with torch.no_grad():
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_compute_time = 0

                start_compute_time = time.perf_counter()
                outputs = model(inputs)

                total_compute_time += time.perf_counter() - start_compute_time
                evaluator.process(inputs, outputs)

            if idx % 20 == 0:
                logger.info('Saving current state of evaluation...')
                save_state(evaluator, evaluator_path)
                save_batch_index(idx, batch_index_path)
            start_idx = idx

            idx += 1
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_batch = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_batch > 30:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / batch. ETA={}".format(
                        idx + 1, total, seconds_per_batch, str(eta)
                    ),
                    n=30,
                )
            # Clean up memory
            # del inputs
            # del outputs
            torch.cuda.empty_cache()

        logger.info('Saving final state of evaluation...')
        save_state(evaluator, evaluator_path)
        save_batch_index(start_idx + 1, batch_index_path)

    model.train(model.training)
    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / batch per device)".format(
            total_time_str, total_time / (total - num_warmup)
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / batch per device)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup)
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.
    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
