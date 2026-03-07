import time

import keras
import numpy as np

import benchmark


class BenchmarkMetricsCallback(keras.callbacks.Callback):
    def __init__(self, start_batch=1):
        super().__init__()
        self.start_batch = start_batch
        self.state = {}
        self.last_batch = None
        self.time_per_step = None

    def _maybe_finalize(self, batch):
        if batch is not None:
            self.last_batch = batch
        if "benchmark_begin" not in self.state:
            return
        self.state["benchmark_end"] = time.time()
        num_steps = self.last_batch - self.state["actual_start_batch"] + 1
        if num_steps > 0:
            self.time_per_step = (
                self.state["benchmark_end"] - self.state["benchmark_begin"]
            ) / num_steps

    # train
    def on_train_begin(self, logs=None):
        self.state = {}
        self.last_batch = None
        self.time_per_step = None

    def on_train_batch_begin(self, batch, logs=None):
        if batch >= self.start_batch and "benchmark_begin" not in self.state:
            self.state["actual_start_batch"] = batch
            self.state["benchmark_begin"] = time.time()

    def on_train_batch_end(self, batch, logs=None):
        self._maybe_finalize(batch)

    def on_train_end(self, logs=None):
        self._maybe_finalize(batch=None)

    # predict
    def on_predict_begin(self, logs=None):
        self.state = {}
        self.last_batch = None
        self.time_per_step = None

    def on_predict_batch_begin(self, batch, logs=None):
        if batch >= self.start_batch and "benchmark_begin" not in self.state:
            self.state["actual_start_batch"] = batch
            self.state["benchmark_begin"] = time.time()

    def on_predict_batch_end(self, batch, logs=None):
        self._maybe_finalize(batch)

    def on_predict_end(self, logs=None):
        self._maybe_finalize(batch=None)


def fit(model, dataset, start_batch=None):
    start_batch = (benchmark.NUM_STEPS // 10) if start_batch is None else start_batch
    callback = BenchmarkMetricsCallback(start_batch=start_batch)
    model.fit(dataset, epochs=1, callbacks=[callback])
    return 1000.0 * callback.time_per_step


def predict(model, dataset, start_batch=None):
    start_batch = (benchmark.NUM_STEPS // 10) if start_batch is None else start_batch
    callback = BenchmarkMetricsCallback(start_batch=start_batch)
    model.predict(dataset, callbacks=[callback])
    return 1000.0 * callback.time_per_step


def generate(model, batch_size, max_length):
    inputs = benchmark.get_prompts(batch_size, benchmark.NUM_WORDS)

    # Build the model by running.
    model.generate(inputs, max_length=max_length)

    # Run another time to get the time of first step and python overhead.
    start_time = time.time()
    model.generate(inputs, max_length=max_length)
    end_time = time.time()
    overhead_time = end_time - start_time

    # Benchmark the running time
    start_time = time.time()
    for _ in range(benchmark.NUM_STEPS + 1):
        model.generate(inputs, max_length=max_length)
    end_time = time.time()
    total_time = end_time - start_time

    return (total_time - overhead_time) / benchmark.NUM_STEPS * 1000


def use_jit():
    # Only use jit_compile=False when using torch backend.
    return not (
        hasattr(keras, "version")
        and keras.version().startswith("3.")
        and keras.backend.backend() == "torch"
    )


def steps_per_execution():
    import os
    return int(os.environ.get("KERAS_STEPS_PER_EXECUTION", 1))


def get_train_dataset_for_text_classification(
    preprocessor, batch_size, seq_len
):
    import tensorflow as tf

    prompts = benchmark.get_prompts(
        num_prompts=batch_size,
        num_words=seq_len,
    )
    dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                tf.constant(prompts),
                tf.constant(np.random.randint(2, size=batch_size)),
            )
        )
        .repeat()
        .batch(batch_size)
        .take(benchmark.NUM_STEPS + 1)
    )

    # Force the dataset to cache into memory.
    dataset = dataset.map(preprocessor).cache()
    count = 0
    for batch in dataset:
        count += 1
    return dataset


def get_train_dataset_for_text_gen(preprocessor, batch_size, seq_len):
    import tensorflow as tf

    prompts = benchmark.get_prompts(
        num_prompts=batch_size,
        num_words=seq_len,
    )
    dataset = (
        tf.data.Dataset.from_tensor_slices(
            (tf.constant(prompts), tf.constant(prompts))
        )
        .repeat()
        .batch(batch_size)
        .take(benchmark.NUM_STEPS + 1)
    )

    # Force the dataset to cache into memory.
    dataset = dataset.map(preprocessor).cache()
    count = 0
    for batch in dataset:
        count += 1
    return dataset
