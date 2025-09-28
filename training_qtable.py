import random
import time
from typing import List, Tuple, Optional

import chess
import chess.engine

from states.QTable import QTable
from states.state_KRvsk import State


class QTableTrainer:
    def __init__(self, epsilon=0.1, alpha=0.5, gamma=0.9, engine_path=r"stockfish\stockfish-windows-x86-64-avx2.exe"):
        self.qtable = QTable()
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.engine_path = engine_path

    def train(self, episodes=1000, initial_fen=None):
        for ep in range(episodes):
            # Initialize chess board and state
            board = chess.Board(initial_fen)
            state = State()
            state.create_from_fen(initial_fen)
            engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
            done = False

            while not done:
                # White (Q-table agent)
                legal_moves = [move.uci() for move in board.legal_moves]
                self.qtable.add_state(state, legal_moves)
                # Epsilon-greedy action selection
                action = self.qtable.argmax_epsilon(state, self.epsilon)
                if action is None:
                    break
                # Make the move on the board
                board.push_san(action)
                # Get new state after move
                new_state = State()
                new_state.create_from_fen(board.fen())

                # Reward placeholder (can be improved)
                if board.is_game_over():
                    done = True
                self.qtable.add_state(new_state, [move.uci() for move in board.legal_moves])

                state_computer = State()
                if not done:
                    # Black (engine)
                    result = engine.play(board, chess.engine.Limit(time=0.1))
                    board.push(result.move)
                    state_computer.create_from_fen(board.fen())

                reward = -0.05
                if board.is_game_over():
                    done = True
                    reward = -100 if board.result() == '1-0' else 10 if board.result() == '0-1' else -10

                # Q-learning update
                best_next = self.qtable.argmax(new_state)
                old_value = self.qtable.q_table[state][action]
                next_value = self.qtable.q_table[new_state][best_next] if best_next else 0
                new_value = old_value + self.alpha * (reward + self.gamma * next_value - old_value)
                self.qtable.set_action_value(state, action, new_value)
                print(f"Episode {ep+1}, State: {state.to_fen()}, Action: {action}, Reward: {reward}, New Value: {new_value}")
                
                if  done:
                    break
                
                state = state_computer

            engine.quit()

    def save_qtable(self, filename):
        self.qtable.save(filename)


# Example usage
if __name__ == "__main__":
    trainer = QTableTrainer(epsilon=0.2, alpha=0.5, gamma=0.9)
    trainer.train(episodes=10, initial_fen="8/8/1k6/8/8/4K3/3R4/8 w - - 0 1")
    trainer.save_qtable("trained_qtable.json")