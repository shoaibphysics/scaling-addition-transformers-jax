"""Fit and visualize Chinchilla-style scaling laws."""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def _huber_loss(residuals, delta):
  """Return elementwise Huber losses."""
  absolute_residuals = np.abs(residuals)

  return np.where(absolute_residuals <= delta,0.5 * residuals**2,delta* (absolute_residuals- 0.5 * delta))


def fit_chinchilla_law(
    parameter_counts,
    training_tokens,
    losses,
    *,
    fixed_floor=None,
    huber_delta=1e-3,
    loss_epsilon=1e-12):
  """Fit L(N, D) = E + A/N**alpha + B/D**beta."""
  parameter_counts = np.asarray(parameter_counts,dtype=np.float64)

  training_tokens = np.asarray(training_tokens,dtype=np.float64)

  losses = np.asarray(losses,dtype=np.float64)

  if not (parameter_counts.shape == training_tokens.shape == losses.shape):
    raise ValueError("N, D, and losses must have matching shapes.")

  parameter_counts = (parameter_counts.reshape(-1))

  training_tokens = (training_tokens.reshape(-1))

  losses = losses.reshape(-1)

  if losses.size < 5:
    raise ValueError("At least five observations are required.")

  if (np.any(~np.isfinite(parameter_counts))
      or np.any(~np.isfinite(training_tokens))
      or np.any(~np.isfinite(losses))):
    raise ValueError("N, D, and losses must be finite.")

  if (np.any(parameter_counts <= 0)
      or np.any(training_tokens <= 0)
      or np.any(losses < 0)):
    raise ValueError("N and D must be positive; losses cannot be negative.")

  if (huber_delta <= 0
      or loss_epsilon <= 0):
    raise ValueError("huber_delta and loss_epsilon must be positive.")

  if np.unique(parameter_counts).size < 3:
    raise ValueError("At least three model sizes are required.")

  if np.unique(training_tokens).size < 3:
    raise ValueError("At least three token budgets are required.")

  clipped_count = int(np.sum(losses < loss_epsilon))

  if clipped_count:
    warnings.warn(f"Clipped {clipped_count} loss value(s) to {loss_epsilon:g} before taking logarithms.",stacklevel=2)

  fitted_losses = np.maximum(losses,loss_epsilon)

  minimum_loss = float(np.min(fitted_losses))

  if fixed_floor is not None:
    fixed_floor = float(fixed_floor)

    if (not np.isfinite(fixed_floor)
        or fixed_floor < 0):
      raise ValueError("fixed_floor must be non-negative and finite.")

  log_n = np.log(parameter_counts)

  log_d = np.log(training_tokens)

  log_losses = np.log(fitted_losses)

  def unpack(theta):
    if fixed_floor is None:
      (log_a,
       log_b,
       log_e,
       alpha,
       beta) = theta

      floor = np.exp(log_e)
    else:
      (log_a,
       log_b,
       alpha,
       beta) = theta

      floor = fixed_floor

    return (log_a,log_b,floor,alpha,beta)

  # Evaluate the positive scaling-law
  # terms stably in logarithmic space.
  def log_predictions(theta):
    (log_a,
     log_b,
     floor,
     alpha,
     beta) = unpack(theta)

    terms = [log_a- alpha * log_n,log_b- beta * log_d]

    if floor > 0:
      terms.append(np.full_like( log_n,np.log(floor)))

    return logsumexp(np.stack(terms),axis=0)

  def objective(theta):
    residuals = (log_predictions(theta) - log_losses)

    return float(np.sum( _huber_loss(residuals,huber_delta)))

  exponent_starts = ((0.1, 0.1),
                     (0.3, 0.3),
                     (0.5, 0.5),
                     (0.8, 0.3),
                     (0.3, 0.8),
                     (1.0, 1.0))

  floor_starts = ((0.1, 0.5, 0.9, 0.99) if fixed_floor is None else (None,))

  starts = []

  for floor_fraction in floor_starts:
    floor = (minimum_loss* floor_fraction if fixed_floor is None else fixed_floor)

    reducible_loss = max(float(np.median(fitted_losses)- floor),loss_epsilon)

    for alpha, beta in exponent_starts:
      log_a = (np.log(reducible_loss / 2.0)+ alpha* np.mean(log_n))

      log_b = (np.log(reducible_loss / 2.0) + beta * np.mean(log_d))

      if fixed_floor is None:
        starts.append([log_a,log_b,np.log(floor),alpha,beta,])
      else:
        starts.append([log_a,log_b,alpha,beta])

  coefficient_bounds = (-100.0,100.0)

  exponent_bounds = (1e-3,2.0)

  if fixed_floor is None:
    lower_floor = max(np.finfo(np.float64).tiny,minimum_loss * 1e-12)

    upper_floor = (minimum_loss* (1.0 - 1e-9))

    bounds = [coefficient_bounds,coefficient_bounds,
              (np.log(lower_floor),np.log(upper_floor)),
              exponent_bounds,
              exponent_bounds]
  else:
    bounds = [coefficient_bounds,coefficient_bounds,exponent_bounds,exponent_bounds]

  solutions = [minimize(objective,np.asarray(start,dtype=np.float64), method="L-BFGS-B",
                        bounds=bounds,
                        options={"maxiter": 5_000,"ftol": 1e-15,"gtol": 1e-10,}) for start in starts]

  finite_solutions = [solution for solution in solutions if np.isfinite(solution.fun)]

  if not finite_solutions:
    raise RuntimeError("Every scaling-law fit failed.")

  best = min(finite_solutions,key=lambda solution: solution.fun)

  (log_a,
   log_b,
   floor,
   alpha,
   beta) = unpack(best.x)

  log_residuals = (log_predictions(best.x) - log_losses)

  exponent_sum = (alpha + beta)

  log_g = (np.log(alpha) + log_a - np.log(beta) - log_b) / exponent_sum

  return {
      "E": float(floor),
      "A": float(np.exp(log_a)),
      "B": float(np.exp(log_b)),
      "alpha": float(alpha),
      "beta": float(beta),
      "G": float(np.exp(log_g)),
      "model_compute_exponent": float(
          beta / exponent_sum),
      "data_compute_exponent": float(
          alpha / exponent_sum),
      "loss_compute_exponent": float(
          alpha
          * beta
          / exponent_sum),
      "huber_objective": float(
          best.fun),
      "rmse_log_loss": float(
          np.sqrt(
              np.mean(
                  log_residuals**2))),
      "floor_was_fixed": (
          fixed_floor is not None),
      "loss_epsilon": float(
          loss_epsilon),
      "num_clipped_losses": (
          clipped_count),
      "num_points": int(
          losses.size),
      "optimizer_success": bool(
          best.success),
      "optimizer_message": str(
          best.message),
  }


def predict_chinchilla_law(
    fit,
    parameter_counts,
    training_tokens):
  """Evaluate a fitted law at one or more (N, D) points."""
  parameter_counts, training_tokens = (
      np.broadcast_arrays(
          np.asarray(
              parameter_counts,
              dtype=np.float64),
          np.asarray(
              training_tokens,
              dtype=np.float64)))

  if (
      np.any(~np.isfinite(parameter_counts))
      or np.any(~np.isfinite(training_tokens))
      or np.any(parameter_counts <= 0)
      or np.any(training_tokens <= 0)):
    raise ValueError(
        "N and D must be positive and finite.")

  terms = [
      np.log(fit["A"])
      - fit["alpha"]
      * np.log(parameter_counts),

      np.log(fit["B"])
      - fit["beta"]
      * np.log(training_tokens),
  ]

  if fit["E"] > 0:
    terms.append(
        np.full_like(
            parameter_counts,
            np.log(fit["E"])))

  return np.exp(
      logsumexp(
          np.stack(terms),
          axis=0))


def compute_optimal_allocation(
    fit,
    training_compute,
    *,
    flop_factor=6.0):
  """Return compute-optimal N and D under C = flop_factor * N * D."""
  training_compute = np.asarray(
      training_compute,
      dtype=np.float64)

  if (
      np.any(~np.isfinite(training_compute))
      or np.any(training_compute <= 0)):
    raise ValueError(
        "training_compute must be positive and finite.")

  if (
      not np.isfinite(flop_factor)
      or flop_factor <= 0):
    raise ValueError(
        "flop_factor must be positive and finite.")

  log_compute = np.log(
      training_compute
      / flop_factor)

  optimal_parameters = np.exp(
      np.log(fit["G"])
      + fit["model_compute_exponent"]
      * log_compute)

  optimal_tokens = np.exp(
      -np.log(fit["G"])
      + fit["data_compute_exponent"]
      * log_compute)

  return (
      optimal_parameters,
      optimal_tokens)


def _record_arrays(records):
  """Extract plotting arrays from saved result records."""
  return {
      "N": np.asarray([
          record["parameter_count"]
          for record in records
      ]),
      "D": np.asarray([
          record["training_token_positions"]
          for record in records
      ]),
      "C": np.asarray([
          record["training_compute_approx"]
          for record in records
      ]),
      "full_loss": np.asarray([
          record["validation"]["full_loss"]
          for record in records
      ]),
      "answer_loss": np.asarray([
          record["validation"]["answer_loss"]
          for record in records
      ]),
      "accuracy": np.asarray([
          record["validation"]["exact_answer_accuracy"]
          for record in records
      ]),
  }


def plot_scaling_summary(
    records,
    full_fit,
    answer_fit,
    *,
    theoretical_full_floor=None,
    output_path=None,
    title="Dense Addition Transformer Scaling Laws"):
  """Plot observed metrics, fitted losses, and optimal frontiers."""
  if not records:
    raise ValueError(
        "records must not be empty.")

  values = _record_arrays(
      records)

  parameter_counts = np.unique(
      values["N"])

  d_curve = np.geomspace(
      np.min(values["D"]),
      np.max(values["D"]),
      200)

  colors = plt.cm.viridis(
      np.linspace(
          0.05,
          0.95,
          parameter_counts.size))

  figure, axes = plt.subplots(
      2,
      2,
      figsize=(12, 9))

  (full_axis,
   answer_axis,
   accuracy_axis,
   frontier_axis) = axes.ravel()

  answer_epsilon = float(
      answer_fit.get(
          "loss_epsilon",
          1e-12))

  for parameter_count, color in zip(
      parameter_counts,
      colors):

    selected = (
        values["N"]
        == parameter_count)

    label = (
        f"{parameter_count / 1e6:.3g}M")

    full_axis.scatter(
        values["D"][selected],
        values["full_loss"][selected],
        color=color,
        s=24)

    full_axis.plot(
        d_curve,
        predict_chinchilla_law(
            full_fit,
            parameter_count,
            d_curve),
        color=color,
        label=label)

    answer_axis.scatter(
        values["D"][selected],
        np.maximum(
            values["answer_loss"][selected],
            answer_epsilon),
        color=color,
        s=24)

    answer_axis.plot(
        d_curve,
        np.maximum(
            predict_chinchilla_law(
                answer_fit,
                parameter_count,
                d_curve),
            answer_epsilon),
        color=color)

    accuracy_axis.scatter(
        values["C"][selected],
        values["accuracy"][selected],
        color=color,
        s=24)

  if theoretical_full_floor is not None:
    full_axis.axhline(
        theoretical_full_floor,
        color="black",
        linestyle="--",
        linewidth=1,
        label="Theoretical floor")

  full_axis.set(
      xlabel="Training token positions, D",
      ylabel="Full-sequence validation NLL",
      title="Full-sequence scaling")

  full_axis.set_xscale(
      "log")

  full_axis.legend(
      title="Parameters",
      fontsize=7)

  answer_axis.set(
      xlabel="Training token positions, D",
      ylabel="Answer-only validation NLL",
      title="Answer-only scaling")

  answer_axis.set_xscale(
      "log")

  answer_axis.set_yscale(
      "log")

  accuracy_axis.set(
      xlabel="Approximate training compute, C = 6ND",
      ylabel="Exact-answer accuracy",
      title="Task accuracy",
      ylim=(-0.02, 1.02))

  accuracy_axis.set_xscale(
      "log")

  frontier_axis.scatter(
      values["N"],
      values["D"],
      color="lightgray",
      edgecolor="gray",
      s=22,
      label="Observed grid")

  compute_curve = np.geomspace(
      np.min(values["C"]),
      np.max(values["C"]),
      500)

  for label, fit, color in (
      (
          "Full NLL frontier",
          full_fit,
          "tab:blue"),
      (
          "Answer NLL frontier",
          answer_fit,
          "tab:orange")):

    optimal_n, optimal_d = (
        compute_optimal_allocation(
            fit,
            compute_curve))

    inside_grid = (
        (optimal_n >= np.min(values["N"]))
        & (optimal_n <= np.max(values["N"]))
        & (optimal_d >= np.min(values["D"]))
        & (optimal_d <= np.max(values["D"])))

    if np.any(inside_grid):
      frontier_axis.plot(
          optimal_n[inside_grid],
          optimal_d[inside_grid],
          color=color,
          linewidth=2,
          label=label)

  frontier_axis.set(
      xlabel="Parameters, N",
      ylabel="Training token positions, D",
      title="Compute-optimal allocation")

  frontier_axis.set_xscale(
      "log")

  frontier_axis.set_yscale(
      "log")

  frontier_axis.legend(
      fontsize=8)

  for axis in axes.ravel():
    axis.grid(
        alpha=0.3)

  figure.suptitle(
      title)

  figure.tight_layout(
      rect=(0, 0, 1, 0.97))

  if output_path is not None:
    output_path = Path(
        output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True)

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight")

  return figure, axes