import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training_qtable_random import QTableTrainer, plot_avg_reward_and_time, plot_training_results
from states.QTable import QTable
from states.state_KRvsk import State



# Test for QTableTrainer training
if __name__ == "__main__":
    qtable = QTable()
    trainer = QTableTrainer(qtable=qtable, epsilon_start=0.2, epsilon_decay=0.999, alpha=0.5, gamma=0.9, min_epsilon=0.1)

    trained_fens = []
    list_of_times = []
    list_of_average_results = []
    num_random_fens = 2
    episodes_per_fen = 10

    try:
        for i in range(num_random_fens):
            fen = State.random_kr_vs_k_fen_column_a()
            print(f"Training on FEN {i+1}: {fen}")
            trained_bool, episode_results, time = trainer.train(episodes=episodes_per_fen, initial_fen=fen)
            if trained_bool:
                print(f"  Successfully trained on FEN: {fen}")
                # Record all symmetric FENs
                for sym_fen, sym_action in State.fen_action_symmetries(fen, None):
                    trained_fens.append(sym_fen)
            #plot_training_results(episode_results)
            list_of_times.append(time)
            average_result = sum(episode_results) / len(episode_results)
            list_of_average_results.append(average_result)

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        # Plot average reward and time
        plot_avg_reward_and_time(list_of_average_results, list_of_times)
        
        # Save the Q-table regardless of interruption
        trainer.save_qtable("results", "test_trained_qtable.json")
        print("Training completed and Q-table saved.")