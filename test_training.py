from training_qtable import QTableTrainer

# Test for QTableTrainer training
if __name__ == "__main__":
    initial_fen = "8/8/1k6/8/8/4K3/3R4/8 w - - 0 1"
    trainer = QTableTrainer( epsilon=0.2, alpha=0.5, gamma=0.9)
    # Run training for a few episodes
    trainer.train(episodes=400, initial_fen=initial_fen)
    # Save the Q-table
    trainer.save_qtable("test_trained_qtable.json")
    print("Training completed and Q-table saved.")
