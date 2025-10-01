import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training_qtable import QTableTrainer
from states.QTable import QTable

# Test for QTableTrainer training
if __name__ == "__main__":
    initial_fen_mate_1 = "8/k1K5/8/8/8/3R4/8/8 w - - 0 1"
    initial_fen_mate_2= "k7/3K4/8/8/8/3R4/8/8 w - - 0 1"
    qtable = QTable()
    trainer = QTableTrainer(qtable=qtable, epsilon_start=0.2, epsilon_decay=0.99, alpha=0.5, gamma=0.9)
    # Run training for a few episodes
    trainer.train(episodes=40, initial_fen=initial_fen_mate_1)
    trainer.train(episodes=450, initial_fen=initial_fen_mate_2)
    # Save the Q-table
    trainer.save_qtable("results", "test_trained_qtable.json")
    print("Training completed and Q-table saved.")

