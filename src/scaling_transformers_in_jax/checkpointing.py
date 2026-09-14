
"""Checkpoint saving and restoration utilities."""

from dataclasses import asdict

import jax
import orbax.checkpoint as ocp


CHECKPOINT_FORMAT_VERSION = 1


# These settings must remain unchanged for an exact continuation.
# num_steps, log_every, num_eval_batches, and checkpoint_every may be changed when resuming.
RESUME_CRITICAL_TRAIN_FIELDS = (
    "batch_size",
    "warmup_steps",
    "learning_rate_decay_step",
    "peak_learning_rate",
    "end_learning_rate",
    "max_grad_norm",
    "weight_decay",
    "adam_epsilon",
    "adam_beta1",
    "adam_beta2")

def create_checkpoint_manager(
    checkpoint_dir,
    max_to_keep=3,
):
  options = ocp.CheckpointManagerOptions(
      max_to_keep=max_to_keep,
      create=True,
      cleanup_tmp_directories=True)

  return ocp.CheckpointManager(
      checkpoint_dir,
      options=options)

def make_run_metadata(
    model_cfg,
    train_cfg,
    vocab,
    split_seed,
    train_size,
    val_size,
    test_size,
):
  return {
      "checkpoint_format_version": (
          CHECKPOINT_FORMAT_VERSION
      ),
      "model_config": asdict(model_cfg),
      "train_config": asdict(train_cfg),
      "data": {
          "vocab": vocab,
          "split_seed": int(split_seed),
          "train_size": int(train_size),
          "val_size": int(val_size),
          "test_size": int(test_size),
      },
  }

def _validate_run_metadata(restored_metadata, expected_metadata):
  restored_version = restored_metadata.get("checkpoint_format_version")
  expected_version = expected_metadata["checkpoint_format_version"]

  if restored_version != expected_version:
    raise ValueError("The checkpoint format version does not match.")

  if (
      restored_metadata.get("model_config")
      != expected_metadata["model_config"]
  ):
    raise ValueError("The model configuration does not match the checkpoint.")

  if (
      restored_metadata.get("data")
      != expected_metadata["data"]
  ):
    raise ValueError(
        "The vocabulary or dataset configuration does not match the checkpoint.")

  restored_train_cfg = restored_metadata.get("train_config",{})

  expected_train_cfg = expected_metadata["train_config"]

  mismatched_fields = [
      field
      for field in RESUME_CRITICAL_TRAIN_FIELDS
      if restored_train_cfg.get(field)
      != expected_train_cfg.get(field)
  ]

  if mismatched_fields:
    raise ValueError(
        "These resume-critical training settings do not match the checkpoint: "
        + ", ".join(mismatched_fields))

def save_checkpoint(
    checkpoint_manager,
    state,
    train_key,
    monitor_key,
    training_history,
    run_metadata):
  step = int(jax.device_get(state.step))

  # Do not try to overwrite a committed or in-progress step.
  if step in checkpoint_manager.all_steps():
    checkpoint_manager.wait_until_finished()

    print(f"Checkpoint at step {step} already exists.")

    return False

  history_snapshot = {
      "records": [
          dict(record)
          for record in training_history
      ]
  }

  saved = checkpoint_manager.save(
      step,
      args=ocp.args.Composite(
          state=ocp.args.StandardSave(state),
          train_key=ocp.args.JaxRandomKeySave(train_key),
          monitor_key=ocp.args.JaxRandomKeySave(monitor_key),
          training_history=ocp.args.JsonSave(history_snapshot),
          run_metadata=ocp.args.JsonSave(dict(run_metadata))
      ),
  )

  checkpoint_manager.wait_until_finished()

  if saved:
    print(f"Saved checkpoint at step {step}.")

  return saved


def restore_latest_checkpoint(
    checkpoint_manager,
    state_template,
    expected_metadata,
):
  checkpoint_manager.wait_until_finished() # Before restoring, we ensure there is no unfinished save operation using the same manager.

  step = checkpoint_manager.latest_step()

  if step is None:
    raise FileNotFoundError("No checkpoint was found.")

  restored = checkpoint_manager.restore(
      step,
      args=ocp.args.Composite(
          state=ocp.args.StandardRestore(state_template),
          train_key=(ocp.args.JaxRandomKeyRestore()),
          monitor_key=(ocp.args.JaxRandomKeyRestore()),
          training_history=(ocp.args.JsonRestore()),
          run_metadata=ocp.args.JsonRestore()
      ),
  )

  _validate_run_metadata(restored.run_metadata, expected_metadata)

  restored_step = int(jax.device_get(restored.state.step))

  if restored_step != step:
    raise ValueError(
        "The restored TrainState step does not match the checkpoint directory.")

  training_history = (restored.training_history["records"])

  print(f"Restored checkpoint at step {restored_step}.")

  return (restored.state,
          restored.train_key,
          restored.monitor_key,
          training_history)
