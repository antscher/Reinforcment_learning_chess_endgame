import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from trainning_mtcs_random import MCTSTrainer, MCTSNode
import chess

# Test for MCTSTrainer training
if __name__ == "__main__":
    trainer = MCTSTrainer()
    trained_fens = []
    list_of_times = []
    list_of_results = []
    num_random_fens = 10
    episodes_per_fen = 2
    max_simulations = 5

    try:
        for i in range(num_random_fens):
            fen = chess.STARTING_FEN
            print(f"Training on FEN {i+1}: {fen}")
            episode_results, time = trainer.train(episodes=episodes_per_fen, initial_fen=fen, max_simulations=max_simulations)
            list_of_times.append(time)
            list_of_results.append(episode_results)
            # Save tree for the first FEN only (for demonstration)
            if i == 0:
                board = chess.Board(fen)
                root = MCTSNode(board)
                for _ in range(max_simulations):
                    node = root
                    while not node.is_terminal() and node.is_fully_expanded():
                        node = node.best_child()
                    if not node.is_terminal():
                        expanded_node = node.expand()
                        if expanded_node is not None:
                            node = expanded_node
                    if node is not None:
                        result = node.rollout()
                        node.backpropagate(result)
                trainer.save_tree(root, "test_mcts_tree.json")

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        print("Training completed and tree saved.")
