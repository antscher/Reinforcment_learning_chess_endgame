import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import time
from typing import List, Tuple, Optional

import chess

from states.QTable import QTable
from states import State
from states.state_KRvsk import State


class QTableTrainer:
    def __init__(self, qtable: QTable, epsilon_start=0.1, epsilon_decay=0.99, alpha=0.5, gamma=0.9, min_epsilon=0.01):
        self.qtable = qtable
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.gamma = gamma
        self.min_epsilon = min_epsilon

    def train(self, episodes=1000, initial_fen=" "):
        reward_mate = 100
        reward_draw = -50
        reward_step = -0.1

        self.epsilon = self.epsilon_start
        start_time = time.time()
        episode_results = []  # Track all results for plotting

        for ep in range(episodes):
            # Initialize chess board and state
            board = chess.Board(initial_fen)
            fen = initial_fen.split(' ')[0]
            nb_moves = 0

            done = False

            while not done:
                nb_moves += 1
                # White (Q-table agent)
                legal_moves = [move.uci() for move in board.legal_moves]
                self.qtable.add_state(fen,  legal_moves)
                # Epsilon-greedy action selection
                action = self.qtable.argmax_epsilon(fen, self.epsilon)
                if action is None:
                    break

                # Make the move on the board
                board.push_san(action)

                # Reward placeholder (can be improved)
                if board.is_game_over():
                    done = True

                #default small negative reward to encourage faster mate
                reward = reward_step
                if not done:
                    # Black (engine)
                    result = random.choice([move.uci() for move in board.legal_moves])  # Random move for black
                    board.push_san(result)

                new_fen = board.fen().split(' ')[0]

                # Check for terminal state
                if board.is_game_over():
                    done = True
                    result = board.result()
                    #print("Game over:", result)
                    reward = reward_mate if result == '1-0' else -reward_mate if result == '0-1' else reward_draw
                    reward /= nb_moves
                    # Track mate results
                    episode_results.append(reward)
                else:
                    # Add new state to Q-table for next iteration
                    self.qtable.add_state(new_fen, [move.uci() for move in board.legal_moves])

                # Q-learning update
                best_next = self.qtable.argmax(new_fen)
                old_value = self.qtable.q_table[fen][action]
                next_value = self.qtable.q_table[new_fen][best_next] if best_next else 0
                new_value = old_value + self.alpha * (reward + self.gamma * next_value - old_value)
                for sym_fen, sym_action in State.fen_action_symmetries(fen, action):
                    if not self.qtable.state_exists(sym_fen):
                        board_sym = chess.Board(sym_fen)
                        self.qtable.add_state(sym_fen, [move.uci() for move in board_sym.legal_moves])
                    self.qtable.set_action_value(sym_fen, sym_action, new_value)
                #print(f"Episode {ep+1}, State: {fen}, Action: {action}, Reward: {reward}, New Value: {new_value}")

                fen = new_fen

                if done:
                    self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
                """
                    # Early stopping: more than 15 mates in last 20 episodes
                    if len(episode_results) >= 20:
                        last_20 = episode_results[-20:]
                        if last_20.count('1-0') > 15:
                            print(f"Early stopping: {last_20.count('1-0')} mates in last 20 episodes.")
                            end_time = time.time()
                            print(f"Training completed in {end_time - start_time:.2f} seconds.")
                            return True, episode_results, end_time - start_time
                    break
                """

        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds.")
        return False, episode_results, end_time - start_time

    def save_qtable(self, folder,filename):
        self.qtable.save(folder,filename)

    def get_qtable(self):
        return self.qtable
    
    def set_qtable(self, qtable: QTable):
        self.qtable = qtable

def plot_training_results(results_list):
    """
    Plot training results for multiple training runs in one plot.
    Each item in results_list is a list of results for one run.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(figsize=(10,5))
    x = np.arange(1, len(results_list)+1)
    plt.plot(x, results_list, label=f'Results')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards per Episode (Combined)')
    plt.legend()
    plt.show()

def plot_avg_reward_and_time(all_results, times):
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
    """
    Plot average reward and execution time for multiple training runs.
    Args:
        all_results (list): List of all training results (rewards) per run.
        times (list): List of execution times (seconds) per run.
    """
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

    # Plot execution time
    plt.figure('Execution Time per Training Run')
    color = 'tab:red'
    plt.plot(runs, times, 's--', color=color, label='Execution Time')
    if len(times) > 3:
        z_time = np.polyfit(runs, times, 3)
        p_time = np.poly1d(z_time)
        plt.plot(runs, p_time(runs), '-', color='green', label='Time Approximation (deg 3)')
    elif len(times) > 2:
        z_time = np.polyfit(runs, times, 2)
        p_time = np.poly1d(z_time)
        plt.plot(runs, p_time(runs), '-', color='green', label='Time Approximation (deg 2)')
    elif len(times) > 1:
        z_time = np.polyfit(runs, times, 1)
        p_time = np.poly1d(z_time)
        plt.plot(runs, p_time(runs), ':', color='orange', label='Time Trend')
    plt.xlabel('Run')
    plt.ylabel('Execution Time (s)')
    plt.title('Execution Time per Training Run')
    plt.legend(loc='upper right')
    plt.tight_layout()

    # Show all figures at once
    plt.show()
