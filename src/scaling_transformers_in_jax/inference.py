"""Load an exported addition Transformer and run greedy inference."""

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp

from .exporting import load_parameters
from .generation import (
    generate_greedy_with_cache,
    make_cached_apply,
)
from .transformer import (
    DecoderOnlyTransformer,
    ModelConfig,
)


__all__ = [
    "AdditionGenerator",
    "DEFAULT_MODEL_DIR",
    "generate_addition",
    "load_addition_generator",
]


DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "addition-dense-v1"
)


@dataclass
class AdditionGenerator:
  """A loaded model with a small prompt-to-answer interface."""

  params: Any
  model_cfg: ModelConfig
  stoi: dict[str, int]
  itos: dict[int, str]
  cached_apply: Callable

  def generate(self, prompt: str) -> str:
    """Continue any in-vocabulary prompt that fits in the context window."""
    if not isinstance(prompt, str):
      raise TypeError("prompt must be a string.")

    if not 1 <= len(prompt) <= self.model_cfg.context_length:
      raise ValueError(
          "Prompt length must be between 1 and "
          f"{self.model_cfg.context_length} characters."
      )

    unknown_characters = sorted(set(prompt) - set(self.stoi))
    if unknown_characters:
      raise ValueError(
          f"Prompt contains unsupported characters: {unknown_characters!r}."
      )

    return generate_greedy_with_cache(
        cached_apply=self.cached_apply,
        params=self.params,
        prompt=prompt,
        stoi=self.stoi,
        itos=self.itos,
        space_id=self.stoi[" "],
        context_length=self.model_cfg.context_length,
    )

  def add(self, a: int, b: int) -> int:
    """Return the integer predicted for a pair of valid operands."""
    if (
        isinstance(a, bool)
        or isinstance(b, bool)
        or not isinstance(a, Integral)
        or not isinstance(b, Integral)
        or not 0 <= a <= 999
        or not 0 <= b <= 999
    ):
      raise ValueError("Both operands must be integers from 0 to 999.")

    prompt = f"{int(a)}+{int(b)}="
    equation = self.generate(prompt)
    answer = equation[len(prompt):]

    if re.fullmatch(r"[0-9]+", answer) is None:
      raise RuntimeError(f"The model produced a malformed equation: {equation!r}")

    return int(answer)

  def __call__(self, prompt: str) -> str:
    return self.generate(prompt)


def _resolve_model_dir(model_dir: str | Path | None) -> Path:
  """Resolve the bundled model by default, with an optional override."""
  if model_dir is None:
    return DEFAULT_MODEL_DIR
  return Path(model_dir).expanduser().resolve()


def _resolve_step(model_dir: Path, step: int | None) -> int:
  """Use the requested export step, or find the latest complete export."""
  if step is not None:
    if isinstance(step, bool) or not isinstance(step, Integral) or step < 0:
      raise ValueError("step must be a non-negative integer or None.")
    return int(step)

  steps = []
  for metadata_path in model_dir.glob("params_step_*.json"):
    step_text = metadata_path.stem.removeprefix("params_step_")
    weights_path = metadata_path.with_suffix(".msgpack")

    if step_text.isdigit() and weights_path.is_file():
      steps.append(int(step_text))

  if not steps:
    raise FileNotFoundError(
        "No complete model export was found in "
        f"{model_dir}. Place the JSON and MessagePack files there, or pass "
        "model_dir=... explicitly."
    )

  return max(steps)


def load_addition_generator(
    model_dir: str | Path | None = None,
    *,
    step: int | None = None,
) -> AdditionGenerator:
  """Load the bundled model, or a model from an explicitly supplied folder."""
  model_dir = _resolve_model_dir(model_dir)
  step = _resolve_step(model_dir, step)

  metadata_path = model_dir / f"params_step_{step}.json"
  weights_path = model_dir / f"params_step_{step}.msgpack"

  if not metadata_path.is_file() or not weights_path.is_file():
    raise FileNotFoundError(
        f"Expected both {metadata_path.name} and {weights_path.name}."
    )

  metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
  model_cfg = ModelConfig(**metadata["model_config"])
  vocab = metadata["vocab"]

  if (
      not isinstance(vocab, str)
      or len(vocab) != model_cfg.vocab_size
      or len(set(vocab)) != len(vocab)
      or set(vocab) != set("0123456789 +=")
  ):
    raise ValueError("The exported vocabulary does not match the model config.")

  model = DecoderOnlyTransformer(cfg=model_cfg)

  # Flax needs the expected parameter structure before deserialization.
  dummy_tokens = jnp.zeros(
      (1, model_cfg.context_length),
      dtype=jnp.int32,
  )
  positions = jnp.arange(model_cfg.context_length, dtype=jnp.int32)
  params_template = model.init(
      jax.random.key(0),
      dummy_tokens,
      positions,
  )["params"]

  params = load_parameters(weights_path, params_template)
  stoi = {character: index for index, character in enumerate(vocab)}
  itos = {index: character for index, character in enumerate(vocab)}

  return AdditionGenerator(
      params=params,
      model_cfg=model_cfg,
      stoi=stoi,
      itos=itos,
      cached_apply=make_cached_apply(model),
  )


@lru_cache(maxsize=4)
def _load_cached_generator(
    model_dir: str,
    step: int,
) -> AdditionGenerator:
  return load_addition_generator(model_dir, step=step)


def generate_addition(
    prompt: str,
    *,
    model_dir: str | Path | None = None,
    step: int | None = None,
) -> str:
  """Continue one prompt, loading and caching the model when first used."""
  model_dir = _resolve_model_dir(model_dir)
  resolved_step = _resolve_step(model_dir, step)

  generator = _load_cached_generator(
      str(model_dir),
      resolved_step,
  )

  return generator.generate(prompt)
