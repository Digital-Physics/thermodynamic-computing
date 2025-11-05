#!/usr/bin/env python3
"""
thermo_ca_fixed.py

Thermodynamic Computing for Cellular Automata Pattern Matching
Uses the thrml library (JAX-based energy-based models with Gibbs sampling)
to find optimal action sequences through thermodynamic processes.

Requires: pip install thrml jax jaxlib matplotlib numpy scipy tqdm
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import thrml
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from tqdm import tqdm
import argparse
import os
import json

plt.ion()

# --- Cellular Automata Environment ---
class CAEnv:
    """Represents the Cellular Automata (CA) environment."""
    def __init__(self, grid_size=12, rules_name='conway',
                 target_pattern=None, max_steps=10):
        self.grid_size = grid_size
        self.rules_name = rules_name
        self.max_steps = max_steps
        self.current_step = 0

        self.ca_rules = {
            'conway': {'birth': [3], 'survive': [2, 3]},
            'seeds': {'birth': [2], 'survive': []},
            'maze': {'birth': [3], 'survive': [1, 2, 3, 4, 5]}
        }
        self.rules = self.ca_rules[self.rules_name]

        self.actions = ['up', 'down', 'left', 'right', 'do_nothing'] + \
                       [f'write_{i:04b}' for i in range(16)]
        self.num_actions = len(self.actions)

        self.ca_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)

        if target_pattern is not None:
            self.target_pattern = target_pattern
        else:
            self.target_pattern = None

        self.reset()

    def _apply_ca_rules_fast(self, grid):
        """Fast CA rule application using convolution."""
        neighbor_counts = convolve2d(grid, self.ca_kernel, mode='same', boundary='wrap')
        birth_mask = np.isin(neighbor_counts, self.rules['birth']) & (grid == 0)
        survive_mask = np.isin(neighbor_counts, self.rules['survive']) & (grid == 1)
        new_grid = np.zeros_like(grid)
        new_grid[birth_mask | survive_mask] = 1
        return new_grid

    def reset(self):
        """Resets the environment to an initial state."""
        self.ca_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.agent_x = self.grid_size // 2
        self.agent_y = self.grid_size // 2
        self.current_step = 0
        return self.ca_grid.copy()

    def step(self, action):
        """Executes one time step in the environment."""
        current_pattern = None

        if 0 <= action <= 3:
            if action == 0: self.agent_y = (self.agent_y - 1 + self.grid_size) % self.grid_size
            elif action == 1: self.agent_y = (self.agent_y + 1) % self.grid_size
            elif action == 2: self.agent_x = (self.agent_x - 1 + self.grid_size) % self.grid_size
            elif action == 3: self.agent_x = (self.agent_x + 1) % self.grid_size
        elif action == 4:
            pass
        elif action >= 5:
            pattern_index = action - 5
            bits = [(pattern_index >> 3) & 1, (pattern_index >> 2) & 1, 
                    (pattern_index >> 1) & 1, pattern_index & 1]
            current_pattern = np.array(bits).reshape(2, 2)

        self._update_ca_fast(current_pattern)
        self.current_step += 1
        done = (self.current_step >= self.max_steps)
        return self.ca_grid.copy(), done

    def _update_ca_fast(self, write_pattern=None):
        """Fast CA update using convolution."""
        self.ca_grid = self._apply_ca_rules_fast(self.ca_grid)
        if write_pattern is not None:
            for i in range(2):
                for j in range(2):
                    y = (self.agent_y + i - 1 + self.grid_size) % self.grid_size
                    x = (self.agent_x + j - 1 + self.grid_size) % self.grid_size
                    self.ca_grid[y, x] = write_pattern[i, j]

    def calculate_fitness(self, final_grid):
        """Calculate fitness based on match with target pattern."""
        if self.target_pattern is None:
            return 0.0
        match_fraction = np.mean(final_grid == self.target_pattern)
        fitness = match_fraction * 100
        if match_fraction == 1.0:
            fitness += 100
        return fitness

    def load_pattern(self, filename):
        """Load pattern from file."""
        if os.path.exists(filename):
            self.target_pattern = np.load(filename)
            print(f"Pattern loaded from {filename}")
            return True
        return False


# --- Thermodynamic Optimizer using THRML ---
class ThermodynamicOptimizer:
    """Uses thrml library for thermodynamic computing with energy-based models."""

    def __init__(self, env, steps=10, 
                 beta_initial=0.1,
                 beta_final=10.0,
                 n_warmup=100,
                 n_samples=1000,
                 steps_per_sample=2):
        self.env = env
        self.steps = steps
        self.num_actions = env.num_actions
        
        self.beta_initial = beta_initial
        self.beta_final = beta_final
        self.beta_current = beta_initial
        
        self.n_warmup = n_warmup
        self.n_samples = n_samples
        self.steps_per_sample = steps_per_sample
        
        self.rng_key = random.PRNGKey(42)
        
        self._build_action_chain()
        
        self.best_sequence = None
        self.best_energy = float('inf')
        self.best_fitness = -float('inf')
        
        self.iteration = 0
        self.beta_history = []
        self.energy_history = []
        self.best_fitness_history = []
        self.mean_energy_history = []
        self.sample_diversity_history = []
        
        self.eval_cache = {}

    def _build_action_chain(self):
        """Build a chain of spin nodes for each action position."""
        self.bits_per_action = int(np.ceil(np.log2(self.num_actions)))
        
        self.nodes = []
        for step_idx in range(self.steps):
            for bit_idx in range(self.bits_per_action):
                node = SpinNode()
                self.nodes.append(node)
        
        self.edges = []
        for i in range(len(self.nodes) - 1):
            self.edges.append((self.nodes[i], self.nodes[i+1]))

    def _spins_to_sequence(self, spin_values):
        """Convert binary spin configuration to action sequence."""
        
        # ROBUST FLATTENING - handle any nested structure
        def to_flat_python_list(data):
            """Convert any nested JAX/numpy/python structure to flat Python list of floats."""
            # Base case: scalar
            if np.isscalar(data):
                return [float(data)]
            
            # Convert JAX/numpy to python first
            if hasattr(data, 'tolist'):
                data = data.tolist()
            
            # Now handle python structures
            if isinstance(data, (int, float)):
                return [float(data)]
            
            if isinstance(data, (list, tuple)):
                result = []
                for item in data:
                    result.extend(to_flat_python_list(item))
                return result
            
            # Fallback for weird types
            try:
                return [float(data)]
            except:
                return [0.0]
        
        # Flatten to python list
        flat_spins = to_flat_python_list(spin_values)
        
        # Ensure correct length
        expected_length = self.steps * self.bits_per_action
        if len(flat_spins) < expected_length:
            flat_spins = flat_spins + [-1.0] * (expected_length - len(flat_spins))
        elif len(flat_spins) > expected_length:
            flat_spins = flat_spins[:expected_length]
        
        # Convert spins to bits
        bits = [int((s + 1.0) / 2.0 + 0.01) for s in flat_spins]  # +0.01 for rounding safety
        
        # Decode to actions
        sequence = []
        for step_idx in range(self.steps):
            start_idx = step_idx * self.bits_per_action
            action_bits = bits[start_idx:start_idx + self.bits_per_action]
            
            action = 0
            for i, bit in enumerate(action_bits):
                action += bit * (2 ** (self.bits_per_action - 1 - i))
            
            action = min(action, self.num_actions - 1)
            sequence.append(action)
        
        return sequence

    def evaluate_sequence_numpy(self, sequence):
        """Evaluate fitness using numpy (for CA simulation)."""
        seq_key = tuple(sequence)
        if seq_key in self.eval_cache:
            return self.eval_cache[seq_key]
        
        self.env.reset()
        for action in sequence:
            _, done = self.env.step(int(action))
            if done:
                break
        
        final_grid = self.env.ca_grid.copy()
        fitness = self.env.calculate_fitness(final_grid)
        energy = -fitness
        
        self.eval_cache[seq_key] = (energy, fitness, final_grid)
        return energy, fitness, final_grid

    def sample_sequences(self, beta, n_samples, n_warmup):
        """Sample action sequences using Gibbs sampling via thrml."""
        
        biases = jnp.zeros(len(self.nodes))
        weights = jnp.ones(len(self.edges)) * 0.3
        
        beta_jax = jnp.array(beta)
        model = IsingEBM(self.nodes, self.edges, biases, weights, beta_jax)
        
        free_blocks = [
            Block(self.nodes[::2]),
            Block(self.nodes[1::2])
        ]
        
        program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
        
        self.rng_key, k_init = random.split(self.rng_key)
        init_state = hinton_init(k_init, model, free_blocks, ())
        
        self.rng_key, k_samp = random.split(self.rng_key)
        schedule = SamplingSchedule(
            n_warmup=n_warmup,
            n_samples=n_samples,
            steps_per_sample=self.steps_per_sample
        )
        
        all_nodes_block = Block(self.nodes)
        
        samples_raw = sample_states(
            k_samp, program, schedule, init_state, [], [all_nodes_block]
        )
        
        samples = []
        energies = []
        
        # Extract samples from return value
        if isinstance(samples_raw, dict):
            state_samples = next(iter(samples_raw.values()))
        else:
            state_samples = samples_raw
        
        # Convert to numpy
        if hasattr(state_samples, '__array__'):
            state_samples = np.array(state_samples)
        
        # Process each sample
        n_actual_samples = min(n_samples, len(state_samples) if hasattr(state_samples, '__len__') else n_samples)
        
        for i in range(n_actual_samples):
            try:
                # Extract spin values for this sample
                if hasattr(state_samples, 'shape') and len(state_samples.shape) > 1:
                    spin_values = state_samples[i]
                else:
                    spin_values = state_samples
                
                # Decode and evaluate
                sequence = self._spins_to_sequence(spin_values)
                energy, fitness, _ = self.evaluate_sequence_numpy(sequence)
                
                samples.append(sequence)
                energies.append(energy)
                
                # Update best
                if energy < self.best_energy:
                    self.best_energy = energy
                    self.best_fitness = fitness
                    self.best_sequence = sequence.copy()
                    
            except Exception as e:
                print(f"\nWarning: Failed to process sample {i}: {e}")
                continue
        
        return samples, energies

    def run_annealing(self, n_iterations=100):
        """Run simulated annealing."""
        print(f"\nRunning thermodynamic annealing with THRML...")
        print(f"Initial beta (1/kT): {self.beta_initial}, Final: {self.beta_final}")
        print(f"Iterations: {n_iterations}\n")
        
        betas = np.logspace(
            np.log10(self.beta_initial),
            np.log10(self.beta_final),
            n_iterations
        )
        
        for iteration in tqdm(range(n_iterations), desc="Annealing"):
            beta = betas[iteration]
            self.beta_current = beta
            
            samples_per_iter = max(10, self.n_samples // 10)
            warmup_per_iter = max(10, self.n_warmup // 10)
            
            samples, energies = self.sample_sequences(
                beta=beta,
                n_samples=samples_per_iter,
                n_warmup=warmup_per_iter
            )
            
            if len(energies) > 0:
                self.beta_history.append(beta)
                self.energy_history.append(np.mean(energies))
                self.mean_energy_history.append(np.mean(energies))
                self.best_fitness_history.append(self.best_fitness)
                
                unique_samples = len(set(tuple(s) for s in samples))
                diversity = unique_samples / len(samples) if len(samples) > 0 else 0
                self.sample_diversity_history.append(diversity)
            
            self.iteration += 1
        
        print(f"\nAnnealing complete!")
        print(f"Best fitness: {self.best_fitness:.2f}")
        print(f"Best energy: {self.best_energy:.2f}")
        
        return self.best_sequence, self.best_fitness


def train_thermodynamic(args):
    """Main thermodynamic training loop."""
    print("=" * 60)
    print("Thermodynamic Computing with THRML")
    print("=" * 60)

    env = CAEnv(grid_size=args.grid_size, rules_name=args.rules, max_steps=args.steps)

    if args.pattern_file and os.path.exists(args.pattern_file):
        env.load_pattern(args.pattern_file)
    else:
        print("Warning: No pattern file loaded.")

    optimizer = ThermodynamicOptimizer(
        env=env,
        steps=args.steps,
        beta_initial=args.beta_initial,
        beta_final=args.beta_final,
        n_warmup=args.n_warmup,
        n_samples=args.n_samples,
        steps_per_sample=args.steps_per_sample
    )

    best_sequence, best_fitness = optimizer.run_annealing(n_iterations=args.iterations)
    
    # Save results
    results = {
        'best_sequence': best_sequence,
        'best_fitness': float(best_fitness),
        'best_energy': float(optimizer.best_energy),
        'beta_history': [float(b) for b in optimizer.beta_history],
        'energy_history': [float(e) for e in optimizer.energy_history],
        'best_fitness_history': [float(f) for f in optimizer.best_fitness_history],
    }
    
    with open('thermodynamic_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    np.save('best_sequence_thermo.npy', np.array(best_sequence))
    
    print(f"\n{'='*60}")
    print(f"Results saved to thermodynamic_results.json and best_sequence_thermo.npy")
    print(f"Best sequence: {best_sequence}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Thermodynamic Computing for CA Pattern Matching (using THRML)")
    
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--n-samples', type=int, default=100)
    parser.add_argument('--n-warmup', type=int, default=50)
    parser.add_argument('--steps-per-sample', type=int, default=2)
    parser.add_argument('--beta-initial', type=float, default=0.1)
    parser.add_argument('--beta-final', type=float, default=10.0)
    parser.add_argument('--grid-size', type=int, default=12)
    parser.add_argument('--rules', type=str, default='conway',
                       choices=['conway', 'seeds', 'maze'])
    parser.add_argument('--pattern-file', type=str, required=True)
    parser.add_argument('--live-plot', action='store_true')
    
    args = parser.parse_args()
    train_thermodynamic(args)