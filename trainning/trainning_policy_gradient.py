import random
import pickle
import math
import numpy as np
import chess
from states.state_KRvsk import State

class Trainer:
    """
    REINFORCE / Policy Gradient trainer for KR vs k.
    White is learning agent; Black plays random.
    Reuses knowledge across episodes and canonical symmetries.
    """
    def __init__(self, gamma=0.9):
        self.policy = {}  # dict[state_id] -> np.array of action probabilities
        self.alpha = 0.01
        self.baseline = 0.0
        self.max_episode_length = 100
        self.temperature = 1.5  # for softmax exploration
        self.rewards_history = []
        self.gamma = gamma  # Discount factor

    def get_canonical_id(self, state):
        # Use FEN board part as canonical ID
        fen = state.to_fen()
        return fen.split(' ')[0]

    def legal_moves(self, state):
        board = chess.Board(state.to_fen())
        return [move for move in board.legal_moves]

    def select_action(self, state):
        """
        Sample an action according to the policy πθ(a|s).
        Return chosen move and canonical state_id.
        Also initializes all symmetries for the state.
        """
        state_id = self.get_canonical_id(state)
        moves = self.legal_moves(state)
        if not moves:
            return None, state_id, []
        # Initialize policy for all symmetries if not present
        symmetries = State.fen_action_symmetries(state_id, None)
        for sym_fen, _ in symmetries:
            if sym_fen not in self.policy:
                self.policy[sym_fen] = np.ones(len(moves)) / len(moves)
        probs = self.policy[state_id]
        # Softmax with temperature
        logits = np.log(probs + 1e-8) / self.temperature
        softmax_probs = np.exp(logits) / np.sum(np.exp(logits))
        action_idx = np.random.choice(len(moves), p=softmax_probs)
        return moves[action_idx], state_id, moves

    def run_episode(self, root_state):
        """
        Play one complete episode from a random root_state.
        Store trajectory: [(state_id, action_index, legal_moves, reward)].
        Return episode reward: 1 (mate) or 0 (draw).
        """
        trajectory = []
        state = State(
            w_king=root_state.get_position('WKing'),
            w_rook=root_state.get_position('WRook'),
            b_king=root_state.get_position('BKing')
        )
        board = chess.Board(state.to_fen())
        for t in range(self.max_episode_length):
            # White's turn
            move, state_id, moves = self.select_action(state)
            if move is None:
                break
            action_idx = moves.index(move)
            # Default step reward (encourage faster mate)
            step_reward = 0
            board.push(move)
            state = State()
            state.create_from_fen(board.fen())
            # Check for mate/draw
            if board.is_game_over():
                result = board.result()
                final_reward = 1 if result == '1-0' else 0
                trajectory.append((state_id, action_idx, moves, final_reward))
                return final_reward, trajectory
            else:
                trajectory.append((state_id, action_idx, moves, step_reward))
            # Black's turn (random)
            black_moves = list(board.legal_moves)
            if not black_moves:
                break
            black_move = random.choice(black_moves)
            board.push(black_move)
            state = State()
            state.create_from_fen(board.fen())
            if board.is_game_over():
                result = board.result()
                final_reward = 1 if result == '1-0' else 0
                # No agent move, so don't append to trajectory
                return final_reward, trajectory
        # Max length reached: treat as draw
        # Only append last step if trajectory is not empty
        if trajectory:
            last_state_id, last_action_idx, last_moves, _ = trajectory[-1]
            trajectory.append((last_state_id, last_action_idx, last_moves, 0))
        return 0, trajectory

    def update_policy(self, episode, final_reward):
        """
        Apply REINFORCE update for each (state, action) in episode using discounted returns:
        θ <- θ + alpha * (G_t - baseline) * grad log πθ(a|s)
        Tabular softmax implementation.
        Symmetry: update all symmetric equivalents using fen_action_symmetries.
        """
        # Compute discounted returns for each step
        rewards = [step[3] for step in episode]
        G = np.zeros(len(rewards))
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = rewards[t] + self.gamma * running_add
            G[t] = running_add
        # Policy update
        for i, (state_id, action_idx, moves, _) in enumerate(episode):
            G_t = G[i]
            symmetries = State.fen_action_symmetries(state_id, None)
            for sym_fen, _ in symmetries:
                if sym_fen not in self.policy:
                    self.policy[sym_fen] = np.ones(len(moves)) / len(moves)
                probs = self.policy[sym_fen]
                grad = -probs
                grad[action_idx] += 1
                self.policy[sym_fen] += self.alpha * (G_t - self.baseline) * grad
                self.policy[sym_fen] = np.clip(self.policy[sym_fen], 1e-8, None)
                self.policy[sym_fen] /= np.sum(self.policy[sym_fen])
        # Update baseline
        self.rewards_history.append(final_reward)
        if len(self.rewards_history) > 100:
            self.rewards_history = self.rewards_history[-100:]
        self.baseline = np.mean(self.rewards_history)

    def train(self, num_episodes):
        for ep in range(num_episodes):
            fen = State.random_kr_vs_k_fen()
            root_state = State()
            root_state.create_from_fen(fen)
            reward, trajectory = self.run_episode(root_state)
            self.update_policy(trajectory, reward)
            print(f"Episode {ep+1}/{num_episodes} | Reward: {reward} | Length: {len(trajectory)} | Baseline: {self.baseline:.3f}")

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.policy, f)
        print(f"Saved policy to {path}")

    def load(self, path):
        with open(path, "rb") as f:
            self.policy = pickle.load(f)
        print(f"Loaded policy from {path}")

def plot_avg_reward_and_time(all_results):
    import matplotlib.pyplot as plt
    def plot_win_stats(results_slice, title):
        wins = sum(1 for r in results_slice if r > 0)
        draws = sum(1 for r in results_slice if r <= 0)
        total_games = len(results_slice)
        win_percentage = 100 * wins / total_games if total_games > 0 else 0
        plt.figure(f'{title} (Win %: {win_percentage:.2f}%)')
        labels = ['Wins', 'Draws']
        values = [wins, draws]
        bars = plt.bar(labels, values, color=['tab:green', 'tab:gray'])
        plt.ylabel('Count')
        plt.title(f'{title} (Win %: {win_percentage:.2f}%)')
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, str(int(yval)), ha='center', va='bottom')
        plt.tight_layout()

    # Flatten all results
    flat_results = [r for results in all_results for r in results]
    n = len(flat_results)
    if n >= 3:
        third = n // 3
        plot_win_stats(flat_results[:third], 'Beginning of Training')
        plot_win_stats(flat_results[third:2*third], 'Middle of Training')
        plot_win_stats(flat_results[2*third:], 'End of Training')
    else:
        plot_win_stats(flat_results, 'Training')

    # Summary plot for all rewards (histogram)
    plt.figure('Summary of All Rewards')
    plt.hist(flat_results, bins=[-0.5, 0.5, 1.5], rwidth=0.8, color='tab:green')
    plt.xticks([0, 1], ['Draw', 'Win'])
    plt.xlabel('Reward')
    plt.ylabel('Count')
    plt.title('Summary of All Rewards')
    plt.tight_layout()

    # Plot wins, draws, and win percentage (summary bar)
    wins = sum(sum(1 for r in results if r > 0) for results in all_results)
    draws = sum(sum(1 for r in results if r <= 0) for results in all_results)
    total_games = sum(len(results) for results in all_results)
    win_percentage = 100 * wins / total_games if total_games > 0 else 0
    plt.figure(f'Wins, Draws, and Win % ({win_percentage:.2f}%)')
    labels = ['Wins', 'Draws']
    values = [wins, draws]
    bars = plt.bar(labels, values, color=['tab:green', 'tab:gray'])
    plt.ylabel('Count')
    plt.title(f'Wins, Draws, and Win % ({win_percentage:.2f}%)')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, str(int(yval)), ha='center', va='bottom')
    plt.tight_layout()
    import numpy as np
    avg_rewards = [sum(results)/len(results)  for results in all_results]
    runs = np.arange(1, len(avg_rewards)+1)
    plt.figure('Average Reward per Training Run')
    color = 'tab:blue'
    plt.plot(runs, avg_rewards, 'o-', color=color, label='Average Reward')
    if len(avg_rewards) > 3:
        z = np.polyfit(runs, avg_rewards, 3)
        p = np.poly1d(z)
        plt.plot(runs, p(runs), '-', color='purple', label='Reward Approximation (deg 3)')
    elif len(avg_rewards) > 2:
        z = np.polyfit(runs, avg_rewards, 2)
        p = np.poly1d(z)
        plt.plot(runs, p(runs), '-', color='purple', label='Reward Approximation (deg 2)')
    elif len(avg_rewards) > 1:
        z = np.polyfit(runs, avg_rewards, 1)
        p = np.poly1d(z)
        plt.plot(runs, p(runs), '--', color='gray', label='Reward Trend')
    plt.xlabel('Run')
    plt.ylabel('Average Reward')
    plt.title('Average Reward per Training Run')
    plt.legend(loc='upper left')
    plt.tight_layout()

    # Show all figures at once
    plt.show()
