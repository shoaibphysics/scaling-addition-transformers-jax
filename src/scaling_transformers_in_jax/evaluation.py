"""Evaluation utilities for the addition experiments."""

import jax
import jax.numpy as jnp

from .training import masked_cross_entropy


def make_answer_mask(inputs, loss_mask, equals_id):
  """Return a mask selecting answer digits and the terminating space."""
  after_equals = (jnp.cumsum((inputs == equals_id).astype(jnp.int32),axis=-1)> 0)

  return loss_mask * after_equals.astype(loss_mask.dtype)


@jax.jit
def evaluate_batch_metrics(state,inputs,targets,loss_mask,equals_id):
  """Return both batch NLLs, their token counts, and exact-answer counts."""
  positions = jnp.arange(inputs.shape[-1],dtype=jnp.int32)

  logits = state.apply_fn({"params": state.params},inputs,positions)

  answer_mask = make_answer_mask(inputs=inputs,loss_mask=loss_mask,equals_id=equals_id)

  full_loss = masked_cross_entropy(logits,targets,loss_mask)

  answer_loss = masked_cross_entropy(logits,targets,answer_mask)

  predictions = jnp.argmax(logits,axis=-1)

  answer_tokens_correct = jnp.logical_or(answer_mask == 0,predictions == targets)

  exact_answer_count = jnp.sum(jnp.all(answer_tokens_correct,axis=-1))

  return (full_loss,
          jnp.sum(loss_mask),
          answer_loss,
          jnp.sum(answer_mask),
          exact_answer_count,
          jnp.asarray(inputs.shape[0], dtype=jnp.int32))


def evaluate_dataset_metrics(state,inputs,targets,loss_mask,equals_id,batch_size=1_000):
  """Evaluate a dataset in batches and return aggregate losses and accuracy."""
  if batch_size <= 0:
    raise ValueError("batch_size must be positive.")

  if inputs.shape[0] == 0:
    raise ValueError("The evaluation dataset must not be empty.")

  full_loss_sum = 0.0
  full_token_count = 0.0

  answer_loss_sum = 0.0
  answer_token_count = 0.0

  exact_answer_count = 0
  example_count = 0

  for start in range(0,inputs.shape[0],batch_size):
    end = min(start + batch_size,inputs.shape[0])

    (batch_full_loss,
     batch_full_count,
     batch_answer_loss,
     batch_answer_count,
     batch_exact_count,
     batch_example_count) = jax.device_get(
             evaluate_batch_metrics(
             state=state,
             inputs=inputs[start:end],
             targets=targets[start:end],
             loss_mask=loss_mask[start:end],
             equals_id=equals_id))

    full_loss_sum += (float(batch_full_loss)* float(batch_full_count))

    full_token_count += float(batch_full_count)

    answer_loss_sum += (float(batch_answer_loss)* float(batch_answer_count))

    answer_token_count += float(batch_answer_count)

    exact_answer_count += int(batch_exact_count)

    example_count += int(batch_example_count)

  return {"full_loss": (full_loss_sum/ full_token_count),
          "answer_loss": (answer_loss_sum/ answer_token_count),
          "exact_answer_accuracy": (exact_answer_count/ example_count),
          "full_token_count": int(full_token_count),
          "answer_token_count": int(answer_token_count),
          "example_count": int(example_count)}