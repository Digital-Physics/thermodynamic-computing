"""
Thermodynamic Computing Demo: Voting with Varying Social Influence

- Each "spin" = a voter (Yes = +1, No = -1)
- Beta controls how strongly voters follow neighbors
- Shows the transition from independent voting to consensus
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

# --- PARAMETERS -------------------------------------------------------------
N = 10             # number of voters
n_samples = 500    # samples per beta
beta_values = [0.5, 1.0, 2.0, 5.0]  # different levels of social influence

# --- CREATE VOTERS AND CONNECTIONS -----------------------------------------
voters = [SpinNode() for _ in range(N)]
edges = [(voters[i], voters[(i + 1) % N]) for i in range(N)]
biases = jnp.zeros(N)
weights = jnp.ones(N) * 0.3

# --- DEFINE SAMPLING BLOCKS -------------------------------------------------
free_blocks = [Block(voters[::2]), Block(voters[1::2])]

# --- INITIALIZE RANDOMLY ----------------------------------------------------
key = jax.random.PRNGKey(0)
k_init, k_samp = jax.random.split(key)
init_state = hinton_init(k_init, IsingEBM(voters, edges, biases, weights, beta_values[0]), free_blocks, ())

# --- LOOP OVER BETA VALUES --------------------------------------------------
plt.figure(figsize=(16, 4))

for i, beta in enumerate(beta_values):
    # Build model for current beta
    model = IsingEBM(voters, edges, biases, weights, beta)
    program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
    schedule = SamplingSchedule(n_warmup=200, n_samples=n_samples, steps_per_sample=2)
    
    # Run sampling
    samples = sample_states(k_samp, program, schedule, init_state, [], [Block(voters)])
    if isinstance(samples, list):
        samples = jnp.stack(samples)
    samples_array = samples.reshape(-1, N)
    
    # Total Yes votes
    total_votes = (samples_array == 1).sum(axis=1)
    
    # --- PLOT HISTOGRAM FOR THIS BETA ---------------------------------------
    plt.subplot(1, len(beta_values), i + 1)
    plt.hist(total_votes.flatten(), bins=range(0, N + 2), color="#38a169", alpha=0.8, align='left')
    plt.title(f"Beta = {beta}")
    plt.xlabel("Yes votes")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)

plt.suptitle("Effect of Social Influence (Beta) on Voting Outcomes")
plt.tight_layout()
plt.show()
