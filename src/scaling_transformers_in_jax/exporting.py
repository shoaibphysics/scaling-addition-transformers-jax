
import json
from dataclasses import asdict
from pathlib import Path

import jax
from flax import serialization


def export_parameters(export_dir, state, model_cfg, vocab):
  # The vocabulary is essential because token IDs depend on its ordering.
  export_dir = Path(export_dir)
  export_dir.mkdir(parents=True, exist_ok=True)

  step = int(jax.device_get(state.step))
  name = f"params_step_{step}"

  params_path = export_dir / f"{name}.msgpack"
  metadata_path = export_dir / f"{name}.json"

  params_path.write_bytes(
      serialization.to_bytes(
          jax.device_get(state.params)
      )
  )

  metadata = {
      "export_format_version": 1, # This is a version number that we define for our own export structure.
      "training_step": step,
      "model_config": asdict(model_cfg),
      "vocab": vocab,
  }

  metadata_path.write_text(
      json.dumps(metadata, indent=2),
      encoding="utf-8",
  )

  print(f"Exported parameters to {params_path}")

  return params_path, metadata_path


def load_parameters(params_path, params_template):
  encoded_parameters = Path(params_path).read_bytes()

  parameters = serialization.from_bytes(
      params_template, # The template tells Flax what kind of structure it should reconstruct.
      encoded_parameters,
  )

  return jax.device_put(parameters)
