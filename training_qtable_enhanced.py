import random
import time
from typing import List, Tuple, Optional

import chess
import chess.engine

from states.QTable import QTable
from states import State
from states.state_KRvsk import State


class QTableTrainer:
    def __init__(self, qtable: QTable, epsilon_start=0.1, epsilon_decay=0.99, alpha=0.5, gamma=0.9, engine_path=r"stockfish\stockfish-windows-x86-64-avx2.exe"):
        self.qtable = qtable
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.gamma = gamma
        self.engine_path = engine_path

    def train(self, episodes=1000, initial_fen=" "):
        reward_mate = 100
        reward_draw = -50
        reward_step = -0.1

        self.epsilon = self.epsilon_start
        engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)

        recent_results = []  # Track last 20 results

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
                    result = engine.play(board, chess.engine.Limit(time=0.1))
                    board.push(result.move)
                    
                new_fen = board.fen().split(' ')[0]

                # Check for terminal state
                if board.is_game_over():
                    done = True
                    result = board.result()
                    print("Game over:", result)
                    reward = reward_mate if result == '1-0' else -reward_mate if result == '0-1' else reward_draw
                    # Track mate results
                    recent_results.append(result)
                else:
                    # Add new state to Q-table for next iteration
                    self.qtable.add_state(new_fen, [move.uci() for move in board.legal_moves])

                # Q-learning update
                best_next = self.qtable.argmax(new_fen)
                old_value = self.qtable.q_table[fen][action]
                next_value = self.qtable.q_table[new_fen][best_next] if best_next else 0
                new_value = old_value + self.alpha * (reward + self.gamma * next_value - old_value)
                # Update Q-table for all symmetries
                for sym_fen, sym_action in State.fen_action_symmetries(fen, action):
                    if not self.qtable.state_exists(sym_fen):
                        board = chess.Board(sym_fen)
                        self.qtable.add_state(sym_fen, [move.uci() for move in board.legal_moves])
                        
                    self.qtable.set_action_value(sym_fen, sym_action, new_value)
                print(f"Episode {ep+1}, State: {fen}, Action: {action}, Reward: {reward}, New Value: {new_value}")

                fen = new_fen
                board = chess.Board(fen)

                # Decay epsilon after each episode
                if done:
                    self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
                    # Early stopping: more than 15 mates in last 20 episodes
                    if len(recent_results) >= 20:
                        last_20 = recent_results[-20:]
                        if last_20.count('1-0') > 15:
                            print(f"Early stopping: {last_20.count('1-0')} mates in last 20 episodes.")
                            engine.quit()
                            return True
                    break

        engine.quit()
        return False

    def save_qtable(self, folder,filename):
        self.qtable.save(folder,filename)

    def get_qtable(self):
        return self.qtable
    
    def set_qtable(self, qtable: QTable):
        self.qtable = qtable

