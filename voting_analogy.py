"""
Thermodynamic Computing Demo: Voting Example

- Each "spin" represents a voter (Yes = +1, No = -1)
- Voters are influenced by neighbors (friends)
- Thermal randomness = independent thinking / stubbornness
- Sampling shows likely voting outcomes across the group
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

# --- PARAMETERS -------------------------------------------------------------
N = 10             # number of voters
beta = 10         # inverse temperature (higher = more social influence)
n_samples = 1000   # number of voting scenarios to sample

# --- CREATE VOTERS AND CONNECTIONS -----------------------------------------
voters = [SpinNode() for _ in range(N)]

# Simple social network: each voter influenced by neighbor
edges = [(voters[i], voters[(i + 1) % N]) for i in range(N)]

# No personal bias (everyone starts neutral)
biases = jnp.zeros(N)

# Weak positive influence: people tend to agree with neighbors
weights = jnp.ones(N) * 0.3

# Build the energy-based voting model
model = IsingEBM(voters, edges, biases, weights, beta)

# --- DEFINE SAMPLING BLOCKS -------------------------------------------------
free_blocks = [Block(voters[::2]), Block(voters[1::2])]
program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])

# --- INITIALIZE RANDOMLY ----------------------------------------------------
key = jax.random.PRNGKey(0)
k_init, k_samp = jax.random.split(key)
init_state = hinton_init(k_init, model, free_blocks, ())

# --- SAMPLING SCHEDULE ------------------------------------------------------
schedule = SamplingSchedule(n_warmup=200, n_samples=n_samples, steps_per_sample=2)

# --- RUN SAMPLING -----------------------------------------------------------
samples = sample_states(k_samp, program, schedule, init_state, [], [Block(voters)])
if isinstance(samples, list):
    samples = jnp.stack(samples)

samples_array = samples.reshape(-1, N)
print("Samples shape:", samples_array.shape)

# --- ANALYSIS ---------------------------------------------------------------
# Total "Yes" votes per sample (sum of +1 votes)
total_votes = (samples_array == 1).sum(axis=1)

# Average vote per voter
avg_votes = np.mean(samples_array, axis=0)

print("Total votes range:", np.min(total_votes), "to", np.max(total_votes))
print("Average votes per voter:", avg_votes)

# --- VISUALIZATION ----------------------------------------------------------

# 1. Histogram: how many voted Yes in each sample
plt.figure(figsize=(8, 4))
plt.hist(total_votes.flatten(), bins=range(0, N + 2), color="#38a169", alpha=0.8, align='left')
plt.title("Distribution of Total 'Yes' Votes")
plt.xlabel("Number of Yes votes")
plt.ylabel("Frequency")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 2. Heatmap: individual votes per sample (Yes = green, No = red)
plt.figure(figsize=(10, 4))
plt.imshow(samples_array.T, cmap="RdYlGn", aspect="auto")
plt.colorbar(label="Vote (+1 = Yes, -1 = No)")
plt.title("Voting Patterns Across Samples")
plt.xlabel("Sample index")
plt.ylabel("Voter index")
plt.tight_layout()
plt.show()

# --- INTERPRETATION ---------------------------------------------------------
"""
At high beta (strong social influence):
    - Most voters agree (all Yes or all No)
    - Histogram peaks at N or 0
    - Heatmap shows uniform color (green or red)

At low beta (weak social influence):
    - Voters act more independently
    - Histogram centers near N/2
    - Heatmap shows mixed colors

Thermodynamic computing here models how a social group
can probabilistically explore possible consensus outcomes,
with "energy" favoring agreement but randomness allowing diversity.
"""
