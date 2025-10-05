import random
import time
from typing import List, Tuple, Optional

import chess
import chess.engine

from states.QTable import QTable
from states import State
from states.state_KRvsk import State


class QTableTrainer:
    def __init__(self, qtable: QTable, epsilon_start=0.1, epsilon_decay=0.99, alpha=0.5, gamma=0.9, engine_path=r"stockfish\stockfish-windows-x86-64-avx2.exe", min_epsilon=0.01):
        self.qtable = qtable
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.gamma = gamma
        self.engine_path = engine_path
        self.min_epsilon = min_epsilon

    def train(self, episodes=1000, initial_fen=" "):
        reward_mate = 100
        reward_draw = -50
        reward_step = -0.1

        self.epsilon = self.epsilon_start
        engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
        start_time = time.time()
        episode_results = []  # Track all results for plotting

        for ep in range(episodes):
            # Initialize chess board and state
            board = chess.Board(initial_fen)
            fen = initial_fen.split(' ')[0]

            done = False

            while not done:
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
                    result = engine.play(board, chess.engine.Limit(time=0.05))
                    board.push(result.move)
                    
                new_fen = board.fen().split(' ')[0]

                # Check for terminal state
                if board.is_game_over():
                    done = True
                    result = board.result()
                    #print("Game over:", result)
                    reward = reward_mate if result == '1-0' else -reward_mate if result == '0-1' else reward_draw
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
                    # Early stopping: more than 15 mates in last 20 episodes
                    if len(episode_results) >= 20:
                        last_20 = episode_results[-20:]
                        if last_20.count('1-0') > 15:
                            print(f"Early stopping: {last_20.count('1-0')} mates in last 20 episodes.")
                            end_time = time.time()
                            print(f"Training completed in {end_time - start_time:.2f} seconds.")
                            engine.quit()
                            return True, episode_results, end_time - start_time
                    break

        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds.")
        engine.quit()
        return False, episode_results, end_time - start_time

    def save_qtable(self, folder,filename):
        self.qtable.save(folder,filename)

    def get_qtable(self):
        return self.qtable
    
    def set_qtable(self, qtable: QTable):
        self.qtable = qtable

def plot_avg_reward_and_time(avg_rewards, times):
    """
    Plot average reward and execution time for multiple training runs.
    Args:
        avg_rewards (list): List of average rewards per run.
        times (list): List of execution times (seconds) per run.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    runs = np.arange(1, len(avg_rewards)+1)
    fig, ax1 = plt.subplots(figsize=(8,5))
    color = 'tab:blue'
    ax1.set_xlabel('Run')
    ax1.set_ylabel('Average Reward', color=color)
    ax1.plot(runs, avg_rewards, 'o-', color=color, label='Average Reward')
    # Improved polynomial curve (approximation) for average reward
    if len(avg_rewards) > 3:
        z = np.polyfit(runs, avg_rewards, 3)
        p = np.poly1d(z)
        ax1.plot(runs, p(runs), '-', color='purple', label='Reward Approximation (deg 3)')
    elif len(avg_rewards) > 2:
        z = np.polyfit(runs, avg_rewards, 2)
        p = np.poly1d(z)
        ax1.plot(runs, p(runs), '-', color='purple', label='Reward Approximation (deg 2)')
    elif len(avg_rewards) > 1:
        z = np.polyfit(runs, avg_rewards, 1)
        p = np.poly1d(z)
        ax1.plot(runs, p(runs), '--', color='gray', label='Reward Trend')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Execution Time (s)', color=color)
    ax2.plot(runs, times, 's--', color=color, label='Execution Time')
    # Improved polynomial curve (approximation) for execution time
    if len(times) > 3:
        z_time = np.polyfit(runs, times, 3)
        p_time = np.poly1d(z_time)
        ax2.plot(runs, p_time(runs), '-', color='green', label='Time Approximation (deg 3)')
    elif len(times) > 2:
        z_time = np.polyfit(runs, times, 2)
        p_time = np.poly1d(z_time)
        ax2.plot(runs, p_time(runs), '-', color='green', label='Time Approximation (deg 2)')
    elif len(times) > 1:
        z_time = np.polyfit(runs, times, 1)
        p_time = np.poly1d(z_time)
        ax2.plot(runs, p_time(runs), ':', color='orange', label='Time Trend')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Average Reward and Execution Time per Training Run')
    fig.tight_layout()
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show()