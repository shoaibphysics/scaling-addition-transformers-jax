"""Autoregressive generation utilities.

At every generation step, the model produces one logit for every token in
the vocabulary. Logits are raw scores rather than probabilities. Applying
softmax converts them into a probability distribution:

    probabilities = softmax(logits)

Greedy decoding:
    Select the token with the largest logit. This is deterministic and is
    the primary decoding method for addition because each problem has one
    correct answer.

Temperature sampling:
    Divide the logits by a positive temperature before applying softmax:

        probabilities = softmax(logits / temperature)

    temperature < 1.0 makes the distribution sharper and more conservative.
    temperature = 1.0 leaves the distribution unchanged.
    temperature > 1.0 makes the distribution flatter and more random.

Top-k sampling:
    Keep exactly the k tokens with the largest logits, remove all other
    candidates, renormalize the remaining probabilities, and sample.

Top-p sampling:
    Also known as nucleus sampling. After temperature scaling and softmax,
    sort the tokens from highest to lowest probability. Keep the smallest
    group of tokens whose probabilities add up to at least p, renormalize
    their probabilities, and sample from that group.

    Because temperature changes the probability distribution, it can also
    change how many tokens are included in the top-p group.

    Unlike top-k, the number of retained tokens adapts to model confidence.

For example, suppose the sorted probabilities are:

    0.50, 0.25, 0.15, 0.07, 0.03

With top-k and k=3, the retained probabilities are:

    0.50, 0.25, 0.15

With top-p and p=0.80, the first three tokens are retained because the first
two sum to only 0.75, while including the third raises the cumulative
probability to 0.90.
"""

import math

import jax
import jax.numpy as jnp


def make_cached_apply(model):
  @jax.jit
  def cached_apply(variables,token_ids,positions,):
    logits, mutable_variables = model.apply(
        variables,
        token_ids,
        positions,
        decode=True,
        mutable=["cache"])

    return logits, mutable_variables["cache"]

  return cached_apply


def _top_k_filter(logits, top_k):
  if top_k is None or top_k == logits.shape[-1]:
    return logits

  top_values, top_indices = jax.lax.top_k(logits,top_k)

  filtered_logits = jnp.full_like(logits,-jnp.inf)

  return filtered_logits.at[top_indices].set(top_values)


def _top_p_filter(logits, top_p):
  if top_p is None or top_p == 1.0:
    return logits

  sorted_indices = jnp.argsort(logits)[::-1]
  sorted_logits = logits[sorted_indices]

  sorted_probabilities = jax.nn.softmax(sorted_logits)

  probability_before = jnp.concatenate(
      (jnp.zeros_like(sorted_probabilities[:1]),jnp.cumsum(sorted_probabilities[:-1])))

  keep = probability_before < top_p

  sorted_logits = jnp.where(
      keep,
      sorted_logits,
      -jnp.inf)

  filtered_logits = jnp.full_like(
      logits,
      -jnp.inf)

  return filtered_logits.at[sorted_indices].set(sorted_logits)


def _greedy_next_token(logits, selection_state):
  token_id = jnp.argmax(logits)

  return token_id, selection_state


def _sample_next_token(
    logits,
    sampling_key,
    temperature,
    top_k,
    top_p):
  filtered_logits = logits.astype(jnp.float32)/ temperature

  filtered_logits = _top_k_filter(filtered_logits,top_k)

  filtered_logits = _top_p_filter(filtered_logits, top_p)

  next_key, sample_key = jax.random.split(sampling_key)

  token_id = jax.random.categorical(
      sample_key,
      filtered_logits,
      axis=-1)

  return token_id, next_key


def _generate_with_cache(
    cached_apply,
    params,
    prompt,
    stoi,
    itos,
    space_id,
    context_length,
    select_next_token,
    selection_state,
    max_new_tokens=None):
  prompt_token_ids = [stoi[character] for character in prompt]

  prompt_length = len(prompt_token_ids)

  if (prompt_length == 0
      or prompt_length > context_length):
    raise ValueError("Prompt length is outside the context window.")

  if max_new_tokens is None:
    max_new_tokens = context_length

  if (not isinstance(max_new_tokens, int)
      or isinstance(max_new_tokens, bool)
      or max_new_tokens <= 0):
    raise ValueError("max_new_tokens must be a positive integer.")

  token_ids = jnp.asarray([prompt_token_ids],dtype=jnp.int32) # [1, prompt_length]

  positions = jnp.arange(prompt_length,dtype=jnp.int32) # [prompt_length]

  # Process the prompt once and initialize the KV cache.
  logits, cache = cached_apply(
      {"params": params},
      token_ids,
      positions)

  output_token_ids = list(prompt_token_ids)

  generated_token_count = 0
  finish_reason = "max_new_tokens"

  while generated_token_count < max_new_tokens:
    next_token_id, selection_state = (
        select_next_token(
            logits[0, -1],
            selection_state))

    next_token_id = int(jax.device_get(next_token_id))

    # The first generated space terminates the answer.
    if next_token_id == space_id:
      finish_reason = "stop"
      break

    if len(output_token_ids) >= context_length:
      finish_reason = "context_length"
      break

    output_token_ids.append(next_token_id)
    generated_token_count += 1
    if generated_token_count >= max_new_tokens:
      break

    position = len(output_token_ids) - 1

    token_ids = jnp.asarray(
        [[next_token_id]],
        dtype=jnp.int32) # [1, 1]

    positions = jnp.asarray(
        [position],
        dtype=jnp.int32) # [1]

    # Process only the newly generated token.
    logits, cache = cached_apply(
        {"params": params,"cache": cache},
        token_ids,
        positions)

  text = "".join(itos[token_id] for token_id in output_token_ids)

  return text, selection_state, finish_reason


def generate_greedy_with_cache(
    cached_apply,
    params,
    prompt,
    stoi,
    itos,
    space_id,
    context_length,
    max_new_tokens=None,
    return_finish_reason=False):
  text, _, finish_reason = _generate_with_cache(
      cached_apply=cached_apply,
      params=params,
      prompt=prompt,
      stoi=stoi,
      itos=itos,
      space_id=space_id,
      context_length=context_length,
      select_next_token=_greedy_next_token,
      selection_state=None,
      max_new_tokens=max_new_tokens)
  if return_finish_reason:
    return text, finish_reason
  
  if finish_reason != "stop":
    raise RuntimeError(f"Greedy generation did not predict the terminating space. Finish reason: {finish_reason!r}. Partial output: {text!r}")

  return text


def generate_sampled_with_cache(
    cached_apply,
    params,
    prompt,
    stoi,
    itos,
    space_id,
    context_length,
    sampling_key,
    temperature=1.0,
    top_k=None,
    top_p=None,
    max_new_tokens=None):
  if (not math.isfinite(temperature)
      or temperature <= 0.0):
    raise ValueError( "temperature must be a positive finite number.")

  vocab_size = len(stoi)

  if (top_k is not None
      and not 1 <= top_k <= vocab_size):
    raise ValueError("top_k must be between 1 and the vocabulary size.")

  if (top_p is not None
      and not 0.0 < top_p <= 1.0):
    raise ValueError("top_p must be in the interval (0, 1].")

  def select_next_token(logits, key):
    return _sample_next_token(
        logits=logits,
        sampling_key=key,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p)

  return _generate_with_cache(
      cached_apply=cached_apply,
      params=params,
      prompt=prompt,
      stoi=stoi,
      itos=itos,
      space_id=space_id,
      context_length=context_length,
      select_next_token=select_next_token,
      selection_state=sampling_key,
      max_new_tokens=max_new_tokens)