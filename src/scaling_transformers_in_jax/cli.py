"""Command-line interface for the exported addition Transformer."""

import argparse
from pathlib import Path

from .inference import generate_addition


def main() -> None:
  parser = argparse.ArgumentParser(
      prog="addition-transformer",
      description="Generate a greedy continuation with the trained model.",
  )
  parser.add_argument("prompt", help="Prompt containing only model-vocabulary characters.")
  parser.add_argument("--model-dir", type=Path, default=None)
  parser.add_argument("--step", type=int, default=None)
  args = parser.parse_args()

  try:
    text = generate_addition(
        args.prompt,
        model_dir=args.model_dir,
        step=args.step,
    )
  except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
    parser.error(str(error))

  print(text)


if __name__ == "__main__":
  main()
