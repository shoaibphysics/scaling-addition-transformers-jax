
"""Training utilities for the addition Transformer."""

import time
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import optax

from .checkpointing import save_checkpoint


@dataclass(frozen=True)
class TrainConfig:
  batch_size: int = 1_024

  log_every: int = 100
  num_eval_batches: int = 4
  checkpoint_every: int = 500

  warmup_steps: int = 500 # End of the warm-up period
  learning_rate_decay_step: int = 10_000 # End of the cosine decay
  num_steps: int = 12_000 # End of this training

  peak_learning_rate: float = 3e-4 # Value obtained at warmup_steps
  end_learning_rate: float = 3e-5 # Value obtained at learning_rate_decay_step and beyond

  max_grad_norm: float = 1.0
  weight_decay: float = 0.1
  adam_epsilon: float = 1e-8
  adam_beta1: float = 0.9
  adam_beta2: float = 0.95


def masked_cross_entropy(logits, targets, loss_mask):
  token_losses = optax.softmax_cross_entropy_with_integer_labels(
      logits = logits,
      labels = targets
  ) # [B,T]

  masked_losses = token_losses * loss_mask # [B,T]

  mean_loss = jnp.sum(masked_losses)/jnp.sum(loss_mask) # scalar

  return mean_loss


def create_optimizer(train_cfg: TrainConfig):
  learning_rate_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
  	peak_value=train_cfg.peak_learning_rate,
  	warmup_steps=train_cfg.warmup_steps,
  	decay_steps=train_cfg.learning_rate_decay_step,
    end_value=train_cfg.end_learning_rate
    )

  optimizer = optax.chain(
    optax.clip_by_global_norm(train_cfg.max_grad_norm),
    optax.adamw(
        learning_rate=learning_rate_schedule,
        b1=train_cfg.adam_beta1,
        b2=train_cfg.adam_beta2,
        eps=train_cfg.adam_epsilon,
        weight_decay=train_cfg.weight_decay,
        )
    )

  return learning_rate_schedule, optimizer


@partial(jax.jit, static_argnames = ("batch_size",))
def train_step(state, key, train_inputs, train_targets, train_masks, batch_size):
  indices = jax.random.randint(
      key,
      shape = (batch_size,),
      minval = 0,
      maxval = train_inputs.shape[0]
  )

  batch_x = train_inputs[indices]
  batch_y = train_targets[indices]
  batch_mask = train_masks[indices]

  positions = jnp.arange(batch_x.shape[-1], dtype = jnp.int32)

  def loss_fn(params):
    logits = state.apply_fn({"params" : params}, batch_x, positions)
    return masked_cross_entropy(logits, batch_y, batch_mask)

  loss, gradients = jax.value_and_grad(loss_fn)(state.params)

  state = state.apply_gradients(grads = gradients)

  return state, loss


@jax.jit
def train_step_with_indices(
    state,
    train_inputs,
    train_targets,
    train_masks,
    indices):
  """Train on one explicitly selected minibatch."""
  batch_x = train_inputs[indices]
  batch_y = train_targets[indices]
  batch_mask = train_masks[indices]

  positions = jnp.arange(
      batch_x.shape[-1],
      dtype=jnp.int32)

  def loss_fn(params):
    logits = state.apply_fn(
        {"params": params},
        batch_x,
        positions)

    return masked_cross_entropy(
        logits,
        batch_y,
        batch_mask)

  loss, gradients = jax.value_and_grad(
      loss_fn)(state.params)

  state = state.apply_gradients(
      grads=gradients)

  return state, loss




@jax.jit
def evaluate_batch_loss(state, inputs, targets, masks):
  positions = jnp.arange(inputs.shape[-1], dtype = jnp.int32)
  logits = state.apply_fn({"params" : state.params}, inputs, positions)
  loss = masked_cross_entropy(logits, targets, masks)
  return loss


@partial(jax.jit, static_argnames = ("batch_size", "num_batches"))
def estimate_dataset_loss(
    state,
    key,
    inputs,
    targets,
    masks,
    batch_size,
    num_batches):
  key, sample_key = jax.random.split(key)

  indices = jax.random.randint(
      sample_key,
      shape = (num_batches, batch_size),
      minval = 0,
      maxval = inputs.shape[0]
  ) # [K, B]

  batch_inputs = inputs[indices] # [K,B,T]
  batch_targets = targets[indices] # [K,B,T]
  batch_masks = masks[indices] # [K,B,T]

  evaluate_batch_losses = jax.vmap(
      evaluate_batch_loss,
      in_axes = (None, 0, 0, 0),
      out_axes = 0)

  batch_losses = evaluate_batch_losses(state, batch_inputs, batch_targets, batch_masks) # [K]

  valid_counts = jnp.sum(batch_masks, axis = (1,2)) # [K]

  mean_loss = jnp.sum(batch_losses * valid_counts) / jnp.sum(valid_counts) # scalar

  return key, mean_loss


def run_training(
    state,
    train_key,
    monitor_key,
    training_data,
    validation_data,
    train_cfg: TrainConfig,
    training_history=None,
    *,
    checkpoint_manager,
    run_metadata,
    training_order=None):
  train_inputs, train_targets, train_masks = training_data
  val_inputs, val_targets, val_masks = validation_data

  if training_history is None:
    training_history = []

  if training_order is not None:
    training_order = jnp.asarray(
        training_order,
        dtype=jnp.int32)

    required_examples = (
        train_cfg.num_steps
        * train_cfg.batch_size)

    if training_order.ndim != 1:
      raise ValueError(
          "training_order must be one-dimensional.")

    if training_order.shape[0] < required_examples:
      raise ValueError(
          "training_order does not contain enough "
          "indices for the requested training horizon.")  

  start_step = int(jax.device_get(state.step))

  if start_step > train_cfg.num_steps:
    raise ValueError(
        f"The state is already at step {start_step}, but num_steps is {train_cfg.num_steps}.")

  latest_checkpoint_step = (checkpoint_manager.latest_step())

  if (
      latest_checkpoint_step is not None
      and latest_checkpoint_step != start_step):
    raise ValueError(
        f"The state is at step {start_step}, but the latest checkpoint is at step {latest_checkpoint_step}.")

  start_time = time.perf_counter()
  interrupted = False

  print(f"Starting training from step {start_step}.")

  try:
    for step in range(start_step,train_cfg.num_steps):
      next_train_key, step_key = jax.random.split(train_key)

      # next_state, _ = train_step(state=state,
      #                            key=step_key,
      #                            train_inputs=train_inputs,
      #                            train_targets=train_targets,
      #                            train_masks=train_masks,
      #                            batch_size=train_cfg.batch_size)

      if training_order is None:
        next_state, _ = train_step(
            state=state,
            key=step_key,
            train_inputs=train_inputs,
            train_targets=train_targets,
            train_masks=train_masks,
            batch_size=train_cfg.batch_size)
      else:
        batch_start = (
            step * train_cfg.batch_size)

        batch_indices = training_order[
            batch_start:
            batch_start + train_cfg.batch_size]

        next_state, _ = train_step_with_indices(
            state=state,
            train_inputs=train_inputs,
            train_targets=train_targets,
            train_masks=train_masks,
            indices=batch_indices)


      # Commit the new state and its matching key together.
      state, train_key = (next_state, next_train_key)

      completed_step = step + 1

      should_log = (completed_step % train_cfg.log_every == 0
                    or completed_step == train_cfg.num_steps)

      if should_log:
        monitor_key, train_loss = estimate_dataset_loss(state=state,
                                                        key=monitor_key,
                                                        inputs=train_inputs,
                                                        targets=train_targets,
                                                        masks=train_masks,
                                                        batch_size=train_cfg.batch_size,
                                                        num_batches=train_cfg.num_eval_batches)
        

        validation_loss = evaluate_batch_loss(state, val_inputs, val_targets, val_masks)

        train_loss, validation_loss = jax.device_get((train_loss,validation_loss))

        elapsed_seconds = time.perf_counter() - start_time

        record = {"step": completed_step,
                  "train_loss": float(train_loss),
                  "validation_loss": float(validation_loss)}
        

        training_history.append(record)

        print(
            f"Step {completed_step:>5} | "
            f"train loss "
            f"{record['train_loss']:.4f} | "
            f"validation loss "
            f"{record['validation_loss']:.4f} | "
            f"time "
            f"{elapsed_seconds / 60:.2f} min"
        )

      should_checkpoint = (completed_step % train_cfg.checkpoint_every == 0
                           and completed_step != train_cfg.num_steps)
          
      if should_checkpoint:
        save_checkpoint(checkpoint_manager=(checkpoint_manager),  
                        state=state,
                        train_key=train_key,
                        monitor_key=monitor_key,
                        training_history=training_history,
                        run_metadata=run_metadata)

  except KeyboardInterrupt:
    interrupted = True

  current_step = int(jax.device_get(state.step))

  # Save the final committed state, whether training
  # completed normally or was manually interrupted.
  save_checkpoint(
      checkpoint_manager=checkpoint_manager,
      state=state,
      train_key=train_key,
      monitor_key=monitor_key,
      training_history=training_history,
      run_metadata=run_metadata)

  elapsed_seconds = time.perf_counter() - start_time

  if interrupted:
    print(f"\nTraining stopped at step {current_step}.")
  else:
    print(f"\nTraining finished at step {current_step}.")

  print(f"Elapsed time: {elapsed_seconds:.2f} seconds ({elapsed_seconds / 60:.2f} minutes)")

  return (state, train_key, monitor_key, training_history)
