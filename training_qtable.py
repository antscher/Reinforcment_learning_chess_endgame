import random
import time
from typing import List, Tuple, Optional

import chess
import chess.engine

from states.QTable import QTable
from states import State


class QTableTrainer:
    def __init__(self, qtable: QTable, epsilon=0.1, alpha=0.5, gamma=0.9, engine_path=r"stockfish\stockfish-windows-x86-64-avx2.exe"):
        self.qtable = qtable
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.engine_path = engine_path

    def train(self, episodes=1000, initial_fen=None):
        engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)

        for ep in range(episodes):
            # Initialize chess board and state
            board = chess.Board(initial_fen)
            fen = initial_fen

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

                reward = -0.05
                if not done:
                    # Black (engine)
                    result = engine.play(board, chess.engine.Limit(time=0.1))
                    board.push(result.move)
                    
                new_fen = board.fen()

                if board.is_game_over():
                    done = True
                    print("Game over:", board.result())
                    reward = 1000 if board.result() == '1-0' else -1000 if board.result() == '0-1' else -100
                
                else:
                    self.qtable.add_state(new_fen, [move.uci() for move in board.legal_moves])

                # Q-learning update
                best_next = self.qtable.argmax(new_fen)
                old_value = self.qtable.q_table[fen][action]
                next_value = self.qtable.q_table[new_fen][best_next] if best_next else 0
                new_value = old_value + self.alpha * (reward + self.gamma * next_value - old_value)
                self.qtable.set_action_value(fen, action, new_value)
                print(f"Episode {ep+1}, State: {fen}, Action: {action}, Reward: {reward}, New Value: {new_value}")
                fen = new_fen
                
                if  done:
                    break

        engine.quit()

    def save_qtable(self, folder,filename):
        self.qtable.save(folder,filename)


