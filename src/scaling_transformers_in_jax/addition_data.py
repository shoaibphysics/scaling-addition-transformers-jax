
import numpy as np
import jax

# Vocabulary
VOCAB = "0123456789 +="
VOCAB_SIZE = len(VOCAB)

stoi = {ch: i for i, ch in enumerate(VOCAB)}
itos = {i: ch for i, ch in enumerate(VOCAB)}

MAX_LEN = len("999+999=1998 ")
SEQ_LEN = MAX_LEN - 1

SPACE = stoi[" "]

# Dataset configuration
SPLIT_SEED = 2026

TRAIN_SIZE = 900_000
VAL_SIZE = 50_000
TEST_SIZE = 50_000

# Encode
def encode(a : int, b : int):
  raw_text = f"{a}+{b}={a+b} "
  stored_text = raw_text.ljust(MAX_LEN, " ")

  token_ids = np.array([stoi[ch] for ch in stored_text],
                       dtype=np.int32)

  inputs = token_ids[:-1]
  targets = token_ids[1:]

  loss_mask = (np.arange(SEQ_LEN) < (len(raw_text) - 1)).astype(np.float32)

  return inputs, targets, loss_mask

# Decode
def decode(token_ids):
  return "".join([itos[int(i)] for i in token_ids])

# Encode with NumPy on the host, then transfer the arrays to the device.
def encode_to_device(pair_pool):
  shape = (len(pair_pool), SEQ_LEN)

  inputs = np.empty(shape, dtype=np.int32)
  targets = np.empty(shape, dtype=np.int32)
  loss_masks = np.empty(shape, dtype=np.float32)

  for i, (a,b) in enumerate(pair_pool):
    inputs[i], targets[i], loss_masks[i] = encode(int(a), int(b))

  device_arrays = jax.device_put((inputs, targets, loss_masks))

  return jax.block_until_ready(device_arrays)

def create_addition_dataset():
  split_rng = np.random.default_rng(SPLIT_SEED)

  all_pairs = np.array(
      [
          (a, b)
          for a in range(1000)
          for b in range(1000)
      ],
      dtype=np.int32,
  )

  split_rng.shuffle(all_pairs)

  val_end = TRAIN_SIZE + VAL_SIZE

  train_pairs = all_pairs[:TRAIN_SIZE]
  val_pairs = all_pairs[TRAIN_SIZE:val_end]
  test_pairs = all_pairs[val_end:] # test_pairs = all_pairs[-TEST_SIZE:]

  train_x, train_y, train_mask = encode_to_device(train_pairs)
  val_x, val_y, val_mask = encode_to_device(val_pairs)
  test_x, test_y, test_mask = encode_to_device(test_pairs)

  # Every split contains:
  # (pairs, inputs, targets, loss mask)
  return {
      "train": (
          train_pairs,
          train_x,
          train_y,
          train_mask,
      ),
      "val": (
          val_pairs,
          val_x,
          val_y,
          val_mask,
      ),
      "test": (
          test_pairs,
          test_x,
          test_y,
          test_mask,
      ),
  }
