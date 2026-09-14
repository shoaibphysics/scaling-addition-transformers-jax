"""Utilities for running resumable dense scaling experiments."""

import json
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import train_state

from .addition_data import SPLIT_SEED, VOCAB, stoi
from .checkpointing import (create_checkpoint_manager,
                            make_run_metadata,
                            restore_latest_checkpoint)
from .evaluation import evaluate_dataset_metrics
from .training import create_optimizer, run_training
from .transformer import DecoderOnlyTransformer


def analytical_dense_parameter_count(model_cfg):
  """Return the parameter count implied by the dense architecture."""
  d_model = model_cfg.d_model

  return int(2 * model_cfg.vocab_size * d_model
             + model_cfg.num_layers
             * (4 * d_model**2+ 3 * d_model * model_cfg.d_ff+ 2 * d_model)
             + d_model)


def count_parameters(params):
  """Count all scalar values in a parameter pytree."""
  return sum(int(parameter.size) for parameter in jax.tree_util.tree_leaves(params))


def make_epochwise_training_order(
    num_examples,
    batch_size,
    num_steps,
    seed):
  """Build deterministic minibatches with one reshuffle per epoch."""
  num_examples = int(num_examples)
  batch_size = int(batch_size)
  num_steps = int(num_steps)

  if (
      num_examples <= 0
      or batch_size <= 0
      or num_steps <= 0):
    raise ValueError(
        "num_examples, batch_size, and "
        "num_steps must be positive.")

  steps_per_epoch = (
      num_examples
      // batch_size)

  if steps_per_epoch == 0:
    raise ValueError(
        "The dataset must contain at least "
        "one complete minibatch.")

  examples_per_epoch = (
      steps_per_epoch
      * batch_size)

  required_examples = (
      num_steps
      * batch_size)

  training_order = np.empty(
      required_examples,
      dtype=np.int32)

  rng = np.random.default_rng(
      int(seed))

  filled = 0

  while filled < required_examples:
    epoch_order = rng.permutation(
        num_examples)

    examples_to_take = min(
        examples_per_epoch,
        required_examples - filled)

    training_order[
        filled:
        filled + examples_to_take
    ] = epoch_order[:examples_to_take]

    filled += examples_to_take

  return training_order



def _validate_run_id(run_id):
  """Require one safe path component for a run ID."""
  if (not isinstance(run_id, str)
      or not run_id
      or Path(run_id).name != run_id
      or run_id in {".", ".."}):
    raise ValueError("run_id must be one non-empty path component.")


def _read_json(path):
  """Read one JSON object."""
  value = json.loads(path.read_text(encoding="utf-8"))

  if not isinstance(value, dict):
    raise ValueError(f"The JSON value in {path} is not an object.")

  return value


def _write_json_atomic(path, value):
  """Write JSON without leaving a partial result file."""
  temporary_path = path.with_name(f".{path.name}.tmp")

  try:
    temporary_path.write_text(json.dumps(value,indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)
  finally:
    temporary_path.unlink(missing_ok=True)


def _validate_identity(saved,expected,source):
  """Prevent a run ID from being reused for another experiment."""
  mismatches = [key for key, value in expected.items() if saved.get(key) != value]

  if mismatches:
    raise ValueError(f"The saved run in {source} does not match: "+ ", ".join(mismatches))


def load_scaling_results(
    *,
    results_dir,
    expected_run_ids,
    expected_protocol=None):
  """Load completed results in grid order and report missing runs."""
  results_dir = Path(results_dir)
  run_ids = list(expected_run_ids)

  if len(run_ids) != len(set(run_ids)):
    raise ValueError("expected_run_ids contains duplicates.")

  records = []
  missing_run_ids = []

  for run_id in run_ids:
    _validate_run_id(run_id)

    result_path = (results_dir / f"{run_id}.json")

    if not result_path.exists():
      missing_run_ids.append(run_id)
      continue

    record = _read_json(result_path)

    if record.get("run_id") != run_id:
      raise ValueError(f"The run ID inside {result_path} is incorrect.")

    if (expected_protocol is not None
        and record.get("protocol")
        != expected_protocol):
      raise ValueError(f"The protocol inside {result_path} is incorrect.")

    records.append(record)

  return records, missing_run_ids


def run_dense_scaling_experiment(
    *,
    protocol,
    run_id,
    model_cfg,
    train_cfg,
    model_seed,
    training_seed,
    dataset,
    checkpoint_root,
    results_dir,
    validation_monitor_size=1_024,
    endpoint_batch_size=1_000,
    max_to_keep=1,
    training_sampling="with_replacement",
    target_training_compute=None):
  """Train or resume one dense run, then save its endpoint metrics."""
  _validate_run_id(run_id)

  if (not isinstance(protocol, str)
      or not protocol):
    raise ValueError(
        "protocol must be a non-empty string.")

  if (train_cfg.num_steps <= 0
      or train_cfg.batch_size <= 0):
    raise ValueError(
        "num_steps and batch_size must be positive.")

  if (validation_monitor_size <= 0
      or endpoint_batch_size <= 0):
    raise ValueError(
        "Evaluation batch sizes must be positive.")

  if training_sampling not in {
      "with_replacement",
      "without_replacement",
      "epoch_shuffle"}:
    raise ValueError(
        "training_sampling must be "
        "'with_replacement', "
        "'without_replacement', or "
        "'epoch_shuffle'.")

  if target_training_compute is not None:
    target_training_compute = float(
        target_training_compute)

    if (
        not np.isfinite(
            target_training_compute)
        or target_training_compute <= 0):
      raise ValueError(
          "target_training_compute must be "
          "positive and finite.")

  try:
    _, train_x, train_y, train_mask = (dataset["train"])

    _, val_x, val_y, val_mask = (dataset["val"])

    _, test_x, _, _ = (dataset["test"])
  except (KeyError,TypeError,ValueError) as error:
    raise ValueError("dataset must have the structure returned by create_addition_dataset().") from error

  if train_x.shape[0] < 2:
    raise ValueError("The training split must contain at least two examples.")

  training_examples_planned = (
      train_cfg.num_steps
      * train_cfg.batch_size)

  steps_per_epoch = int(
      train_x.shape[0]
      // train_cfg.batch_size)

  examples_per_epoch = (
      steps_per_epoch
      * train_cfg.batch_size)

  examples_dropped_per_epoch = (
      int(train_x.shape[0])
      - examples_per_epoch)

  if training_sampling == "without_replacement":
    if training_examples_planned > train_x.shape[0]:
      raise ValueError(
          "Without-replacement training requires "
          f"{training_examples_planned:,} examples, "
          "but the training split contains only "
          f"{train_x.shape[0]:,}.")

    training_order = np.random.default_rng(
        int(training_seed)).permutation(
            train_x.shape[0]).astype(
                np.int32)

  elif training_sampling == "epoch_shuffle":
    training_order = (
        make_epochwise_training_order(
            num_examples=train_x.shape[0],
            batch_size=train_cfg.batch_size,
            num_steps=train_cfg.num_steps,
            seed=training_seed))

  else:
    training_order = None

  for split_name, inputs in (
      ("train", train_x),
      ("val", val_x),
      ("test", test_x)):

    if (inputs.ndim != 2
        or inputs.shape[-1]
        != model_cfg.context_length):
      raise ValueError(f"The {split_name} split does not match the model context length.")

  data_description = {
      "vocab": VOCAB,
      "split_seed": int(SPLIT_SEED),
      "train_size": int(train_x.shape[0]),
      "val_size": int(val_x.shape[0]),
      "test_size": int(test_x.shape[0]),
  }

  if training_order is not None:
    data_description.update({
        "training_sampling": training_sampling,
        "data_order_seed": int(training_seed),
    })


  if training_sampling == "epoch_shuffle":
    data_description.update({
        "steps_per_epoch": steps_per_epoch,
        "examples_per_epoch": (
            examples_per_epoch),
        "examples_dropped_per_epoch": (
            examples_dropped_per_epoch),
    })

  run_identity = {
      "protocol": protocol,
      "run_id": run_id,
      "model_config": asdict(model_cfg),
      "train_config": asdict(train_cfg),
      "model_seed": int(model_seed),
      "training_seed": int(training_seed),
      "data": data_description,
  }

  if target_training_compute is not None:
    run_identity[
        "target_training_compute"
    ] = target_training_compute

  checkpoint_dir = (Path(checkpoint_root)/ run_id)

  results_dir = Path(results_dir)

  checkpoint_dir.mkdir(parents=True,exist_ok=True)

  results_dir.mkdir(parents=True,exist_ok=True)

  result_path = (results_dir/ f"{run_id}.json")

  # A result file marks a completed endpoint.
  if result_path.exists():
    result = _read_json(result_path)

    _validate_identity(result,run_identity,result_path)

    print("Result already exists:",result_path)

    return result

  identity_path = (checkpoint_dir/ "run_identity.json")

  if identity_path.exists():
    _validate_identity(
        _read_json(identity_path),
        run_identity,
        identity_path)
  else:
    _write_json_atomic(identity_path,run_identity)

  model = DecoderOnlyTransformer(cfg=model_cfg)

  positions = jnp.arange(model_cfg.context_length,dtype=jnp.int32)

  model_variables = model.init(jax.random.key(int(model_seed)),train_x[:2],positions)

  _, optimizer = create_optimizer(train_cfg)

  state_template = (
      train_state.TrainState.create(
          apply_fn=model.apply,
          params=model_variables["params"],
          tx=optimizer))

  parameter_count = count_parameters(state_template.params)

  analytical_count = (analytical_dense_parameter_count(model_cfg))

  if parameter_count != analytical_count:
    raise ValueError(
        "The analytical and parameter-tree counts "
        "do not match: "
        f"{analytical_count:,} versus "
        f"{parameter_count:,}.")

  checkpoint_manager = (
      create_checkpoint_manager(
          checkpoint_dir=checkpoint_dir,
          max_to_keep=max_to_keep))

  run_metadata = make_run_metadata(
      model_cfg=model_cfg,
      train_cfg=train_cfg,
      vocab=VOCAB,
      split_seed=SPLIT_SEED,
      train_size=train_x.shape[0],
      val_size=val_x.shape[0],
      test_size=test_x.shape[0])

  if training_order is not None:
    run_metadata["data"].update({
        "training_sampling": training_sampling,
        "data_order_seed": int(training_seed),
    })

  if training_sampling == "epoch_shuffle":
    run_metadata["data"].update({
        "steps_per_epoch": steps_per_epoch,
        "examples_per_epoch": (
            examples_per_epoch),
        "examples_dropped_per_epoch": (
            examples_dropped_per_epoch),
    })


  monitor_size = min(validation_monitor_size,val_x.shape[0])

  try:
    if checkpoint_manager.latest_step() is None:
      state = state_template

      train_key, monitor_key = (jax.random.split(jax.random.key(int(training_seed))))

      training_history = []

      print("Starting a new run:",run_id)
    else:
      (state,
       train_key,
       monitor_key,
       training_history) = (
          restore_latest_checkpoint(
              checkpoint_manager=checkpoint_manager,
              state_template=state_template,
              expected_metadata=run_metadata))

    (state,
     train_key,
     monitor_key,
     training_history) = run_training(
         state=state,
         train_key=train_key,
         monitor_key=monitor_key,
         training_data=(
             train_x,
             train_y,
             train_mask),
         validation_data=(
             val_x[:monitor_size],
             val_y[:monitor_size],
             val_mask[:monitor_size]),
         train_cfg=train_cfg,
         training_history=training_history,
         checkpoint_manager=checkpoint_manager,
         run_metadata=run_metadata,
         training_order=training_order)

    completed_steps = int(jax.device_get(state.step))

    # An interrupted run remains resumable,
    # but it is not treated as an endpoint.
    if completed_steps != train_cfg.num_steps:
      print(
          f"Run paused at step "
          f"{completed_steps:,} of "
          f"{train_cfg.num_steps:,}. "
          "No result was written.")

      return None

    validation_metrics = (
        evaluate_dataset_metrics(
            state=state,
            inputs=val_x,
            targets=val_y,
            loss_mask=val_mask,
            equals_id=stoi["="],
            batch_size=endpoint_batch_size))

    training_examples = (completed_steps* train_cfg.batch_size)

    training_token_positions = (training_examples * model_cfg.context_length)

    mean_scored_targets = float(
        jax.device_get(
            jnp.mean(
                jnp.sum(
                    train_mask,
                    axis=-1))))

    if training_order is None:
      consumed_indices = None

      scored_training_tokens = int(
          round(
              training_examples
              * mean_scored_targets))

      unique_training_examples = None

    else:
      consumed_indices = np.asarray(
          training_order[
              :training_examples])

      scored_per_example = np.asarray(
          jax.device_get(
              jnp.sum(
                  train_mask,
                  axis=-1)))

      scored_training_tokens = int(
          np.sum(
              scored_per_example[
                  consumed_indices],
              dtype=np.int64))

      if (
          training_sampling
          == "without_replacement"):
        unique_training_examples = (
            training_examples)
      else:
        unique_training_examples = int(
            np.unique(
                consumed_indices).size)

    training_compute_approx = (
        6
        * parameter_count
        * training_token_positions)

    if target_training_compute is None:
      relative_compute_error = None
    else:
      relative_compute_error = float(
          (
              training_compute_approx
              - target_training_compute
          )
          / target_training_compute)

    if training_sampling == "epoch_shuffle":
      effective_epochs = (
          completed_steps
          / steps_per_epoch)
    else:
      effective_epochs = None

    
    result = {
        **run_identity,
        "parameter_count": parameter_count,
        "optimizer_steps": completed_steps,
        "training_examples": training_examples,
        "training_token_positions": (training_token_positions),
        "mean_scored_targets_per_example": ( mean_scored_targets),
        "expected_scored_training_tokens": ( training_examples* mean_scored_targets),
        "scored_training_tokens": ( scored_training_tokens),
        "unique_training_examples": (
            unique_training_examples),
        "nominal_training_passes": (
            training_examples
            / train_x.shape[0]),
        "steps_per_epoch": (
            steps_per_epoch
            if training_sampling
            == "epoch_shuffle"
            else None),
        "effective_epochs": (
            effective_epochs),
        "training_compute_approx": (
            training_compute_approx),
        "relative_compute_error": (
            relative_compute_error),
        "validation": validation_metrics,
        "validation_monitor_size": monitor_size,
        "endpoint_batch_size": endpoint_batch_size,
        "training_history": training_history,
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "checkpoint_dir": str(checkpoint_dir)}

    _write_json_atomic(result_path,result)

    print("Saved endpoint result to:",result_path)

    return result
  finally:
    checkpoint_manager.close()