"""
Thermodynamic Computing Demo with `thrml`

This script demonstrates how thermodynamic computing uses
stochastic sampling, similar to how physical systems evolve,
to perform computations by minimizing "energy" which can be though of as error.

We’ll use a simple Ising model:
    - N binary spins (each can be +1 or -1)
    - Spins prefer to align (lower energy when equal)
    - Temperature controls randomness (higher T = more disorder)

Conceptually, this connects to:
- Central Limit Theorem: The sum/average of many spins can approximate a Gaussian at high temperature.
- Sampling: We draw configurations from a Boltzmann-like distribution: P(x) ∝ exp(-βE(x)).
- Thermodynamic computing: Using physical sampling dynamics to explore low-energy solutions.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# Correct thrml imports
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

# --- PARAMETERS -------------------------------------------------------------
N = 10             # number of spins
beta = 0.8         # inverse temperature (higher = more ordered)
n_samples = 1000   # number of configurations to sample

# --- CREATE SPINS AND CONNECTIVITY -----------------------------------------
# Each spin node
nodes = [SpinNode() for _ in range(N)]

# Connect spins in a simple ring (neighbors interact)
edges = [(nodes[i], nodes[(i + 1) % N]) for i in range(N)]

# No individual bias
biases = jnp.zeros(N)

# Weak positive coupling (neighboring spins like to align)
weights = jnp.ones(N) * 0.3

# Build the Ising energy-based model
model = IsingEBM(nodes, edges, biases, weights, beta)

# --- DEFINE BLOCKS FOR SAMPLING --------------------------------------------
# Use two alternating blocks for Gibbs sampling
free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]

# Sampling program
program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])

# --- INITIALIZE RANDOMLY ----------------------------------------------------
key = jax.random.PRNGKey(0)
k_init, k_samp = jax.random.split(key)

# Hinton initialization gives a valid starting state
init_state = hinton_init(k_init, model, free_blocks, ())

# --- SAMPLING SCHEDULE ------------------------------------------------------
schedule = SamplingSchedule(n_warmup=200, n_samples=n_samples, steps_per_sample=2)

# --- RUN SAMPLING -----------------------------------------------------------
samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])

# Stack if returned as list
if isinstance(samples, list):
    samples = jnp.stack(samples)

print("Samples shape:", samples.shape)  # Expect (1, n_samples, N) or (n_samples, N)

# --- ANALYSIS ---------------------------------------------------------------
# Flatten samples if needed
samples_array = samples.reshape(-1, N)

# Total magnetization per sample
total_magnetization = np.sum(np.array(samples_array), axis=1)

# Average spin per site
avg_spin_values = np.mean(np.array(samples_array), axis=0)

print("Magnetization range:", np.min(total_magnetization), "to", np.max(total_magnetization))
print("Mean magnetization:", np.mean(total_magnetization))
print("Average spin per site:", avg_spin_values)

# --- VISUALIZATION ----------------------------------------------------------

# 1. Histogram of total magnetization
plt.figure(figsize=(8, 4))
plt.hist(total_magnetization.flatten(), bins=50, color="#38a169", alpha=0.8)
plt.title("Distribution of Total Magnetization")
plt.xlabel("Magnetization (sum of spins per sample)")
plt.ylabel("Frequency")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 2. Heatmap of average spin per site
plt.figure(figsize=(10, 4))
plt.imshow(samples_array.T, cmap="coolwarm", aspect="auto")
plt.colorbar(label="Spin value (-1 = blue, +1 = red)")
plt.title("Spin Configurations Across Samples")
plt.xlabel("Sample index")
plt.ylabel("Spin index")
plt.tight_layout()
plt.show()

# --- INTERPRETATION ---------------------------------------------------------
"""
At higher beta (lower temperature):
    - Spins align (all +1 or all -1)
    - Histogram shows peaks near ±N
    - Heatmap mostly solid red or blue

At lower beta (higher temperature):
    - Spins fluctuate randomly
    - Histogram centers around 0
    - Heatmap shows mixed colors

Thermodynamic computing leverages these natural sampling dynamics
to find low-energy states or perform probabilistic computation.
"""
