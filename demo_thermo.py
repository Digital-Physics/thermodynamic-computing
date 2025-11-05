#!/usr/bin/env python3
"""
demo_thermo.py

Demonstrate the best sequence found by thermodynamic optimization.
Visualizes the CA evolution step-by-step with animation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import convolve2d
import argparse
import os

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
        self.target_pattern = target_pattern
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


def create_action_images(num_actions, size=20):
    """Pre-generate action visualization images."""
    action_images = {}
    
    # Arrows
    inner_size = size - 4
    img = np.zeros((inner_size, inner_size))
    mid = inner_size // 2
    head_size = inner_size // 3
    
    # Up
    for i in range(head_size):
        img[i, mid - i : mid + i + 1] = 1
    img[head_size:, mid - 1 : mid + 2] = 1
    action_images[0] = np.pad(img, 2, 'constant', constant_values=0)
    
    # Down, Left, Right
    action_images[1] = np.flipud(action_images[0])
    action_images[2] = np.rot90(action_images[0], 1)
    action_images[3] = np.rot90(action_images[0], -1)
    
    # Do nothing (empty set symbol)
    img = np.zeros((size, size))
    center = size // 2
    radius = size // 3 + 1
    yy, xx = np.ogrid[-center:size-center, -center:size-center]
    dist_sq = xx*xx + yy*yy
    circle_mask = (dist_sq <= radius*radius) & (dist_sq >= (radius-2)**2)
    y_idx, x_idx = np.indices((size, size))
    line_mask = abs(y_idx - x_idx) < 2
    img[circle_mask | line_mask] = 1
    action_images[4] = img
    
    # Write patterns
    for i in range(16):
        action_id = i + 5
        bits = [(i >> 3) & 1, (i >> 2) & 1, (i >> 1) & 1, i & 1]
        pattern = np.array(bits).reshape(2, 2)
        cell_size = (size - 2) // 2
        pattern_img = np.kron(pattern, np.ones((cell_size, cell_size)))
        pattern_img = np.pad(pattern_img, 1, 'constant', constant_values=0)
        action_images[action_id] = pattern_img
    
    return action_images


def run_demo(args):
    """Demonstrate a saved action sequence."""
    print("=" * 60)
    print("Thermodynamic Sequence Demo")
    print("=" * 60)

    if not os.path.exists(args.sequence_file):
        raise FileNotFoundError(f"Sequence file not found: {args.sequence_file}")

    sequence = np.load(args.sequence_file)
    print(f"Loaded sequence of length {len(sequence)}")
    print(f"Sequence: {sequence}")

    env = CAEnv(grid_size=args.grid_size, rules_name=args.rules,
                max_steps=len(sequence))
    
    action_images = create_action_images(env.num_actions, size=20)

    if args.pattern_file and os.path.exists(args.pattern_file):
        env.load_pattern(args.pattern_file)

    # Setup visualization
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[4, 4, 1], hspace=0.4, wspace=0.3)

    ax_grid = fig.add_subplot(gs[0, 0])
    ax_target = fig.add_subplot(gs[0, 1])
    ax_actions = fig.add_subplot(gs[1, 0])
    ax_info = fig.add_subplot(gs[1, 1])
    ax_seq_imgs = fig.add_subplot(gs[2, :])

    fig.suptitle('Thermodynamic Sequence Demonstration', fontsize=14, fontweight='bold')

    # Target pattern
    if env.target_pattern is not None:
        ax_target.imshow(env.target_pattern, cmap='binary', vmin=0, vmax=1, interpolation='nearest')
    ax_target.set_title('Target Pattern', fontweight='bold')
    ax_target.set_xticks([])
    ax_target.set_yticks([])

    # Grid
    state = env.reset()
    grid_img = ax_grid.imshow(state, cmap='binary', vmin=0, vmax=1, interpolation='nearest')
    agent_patch = plt.Rectangle((env.agent_x - 1.5, env.agent_y - 1.5), 2, 2,
                                facecolor='none', edgecolor='cyan', linewidth=2)
    ax_grid.add_patch(agent_patch)
    title_text = ax_grid.set_title("Step 0", fontweight='bold')
    ax_grid.set_xticks([])
    ax_grid.set_yticks([])

    # Action sequence bar chart
    action_labels = ['↑', '↓', '←', '→', '∅'] + [f'{i:X}' for i in range(16)]
    action_colors = plt.cm.tab20(np.linspace(0, 1, len(action_labels)))
    colors = [action_colors[a] for a in sequence]
    bars = ax_actions.bar(range(len(sequence)), sequence, color=colors)
    ax_actions.set_title('Action Sequence', fontweight='bold')
    ax_actions.set_xlabel('Step')
    ax_actions.set_ylabel('Action ID')
    ax_actions.set_ylim([-1, env.num_actions])
    ax_actions.grid(axis='y', alpha=0.3)

    # Info
    info_text = ax_info.text(0.05, 0.95, '', va='top', fontsize=11, family='monospace')
    ax_info.set_title('Info', fontweight='bold')
    ax_info.axis('off')

    # Action image sequence
    ax_seq_imgs.set_title('Action History', fontweight='bold')
    ax_seq_imgs.axis('off')

    def update(frame):
        # Reset environment at the start of each loop
        if frame == 0:
            env.reset()
            grid_img.set_data(env.ca_grid)
            agent_patch.set_xy((env.agent_x - 1.5, env.agent_y - 1.5))
            title_text.set_text("Step 0")

            # Reset bar colors
            for i, bar in enumerate(bars):
                bar.set_color(colors[i])
                bar.set_alpha(1.0)
            
            # Clear image sequence
            ax_seq_imgs.clear()
            ax_seq_imgs.set_title('Action History', fontweight='bold')
            ax_seq_imgs.axis('off')

            info_text.set_text("Step: 0\nStarting...")

        elif frame <= len(sequence):
            action = sequence[frame - 1]
            state, _ = env.step(action)

            grid_img.set_data(state)
            agent_patch.set_xy((env.agent_x - 1.5, env.agent_y - 1.5))
            title_text.set_text(f"Step {frame}")

            # Highlight current action
            for i, bar in enumerate(bars):
                if i == frame - 1:
                    bar.set_color('red')
                    bar.set_alpha(1.0)
                elif i < frame - 1:
                    bar.set_color(colors[i])
                    bar.set_alpha(0.3)
                else:
                    bar.set_color(colors[i])
                    bar.set_alpha(1.0)

            # Update action image sequence
            current_sequence = sequence[:frame]
            if len(current_sequence) > 0:
                seq_img_list = [action_images[act] for act in current_sequence]
                padding = np.zeros_like(seq_img_list[0][:, :2])
                padded_imgs = [img for act_img in seq_img_list for img in (act_img, padding)][:-1]
                composite_img = np.hstack(padded_imgs)
                ax_seq_imgs.clear()
                ax_seq_imgs.set_title('Action History', fontweight='bold')
                ax_seq_imgs.axis('off')
                ax_seq_imgs.imshow(composite_img, cmap='binary', interpolation='nearest', aspect='equal')

            fitness = env.calculate_fitness(state)
            match_pct = np.mean(state == env.target_pattern) if env.target_pattern is not None else 0

            info_str = (
                f"Step: {frame}/{len(sequence)}\n"
                f"Action: {action_labels[action]} (ID: {action})\n"
                f"Fitness: {fitness:.2f}\n"
                f"Match: {match_pct:.1%}\n"
                f"Alive cells: {np.sum(state)}\n"
                f"Agent pos: ({env.agent_x}, {env.agent_y})"
            )
            if frame == len(sequence):
                info_str += f"\n\n✓ Final Score: {fitness:.2f}"
                if match_pct == 1.0:
                    info_str += "\n✓ PERFECT MATCH!"

            info_text.set_text(info_str)

        return [grid_img, agent_patch] + list(bars) + [info_text]

    # Add pause frames at the end before looping
    total_frames = len(sequence) + 1 + 30  # 0 (reset) + sequence steps + 30 pause frames

    ani = animation.FuncAnimation(fig, update, frames=total_frames,
                                 interval=500, repeat=True, blit=False)
    
    print("\nPlaying animation... (Press Ctrl+C to stop)")
    plt.show(block=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo thermodynamic sequence")
    
    parser.add_argument('--sequence-file', type=str, default='best_sequence_thermo.npy',
                       help='Sequence file to demonstrate')
    parser.add_argument('--pattern-file', type=str, default='target_pattern.npy',
                       help='Target pattern file')
    parser.add_argument('--grid-size', type=int, default=12,
                       help='Size of CA grid')
    parser.add_argument('--rules', type=str, default='conway',
                       choices=['conway', 'seeds', 'maze'],
                       help='CA rules to use')
    
    args = parser.parse_args()
    run_demo(args)