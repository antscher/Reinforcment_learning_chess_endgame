import torch
import random

"""
Recommended: Use a conda environment for PyTorch GPU support.
Setup:
    conda create -n chess_gpu python=3.11
    conda activate chess_gpu
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    pip install python-chess
"""

# --- Imports ---
import torch
import random
import chess
from states.state_KRvsk import State

# --- Helper Functions (stubs for GPU Q-learning) ---
def encode_state(fen):
    # Example: encode FEN to integer index (hash or lookup)
    # For demonstration, use hash modulo table size
    return abs(hash(fen)) % q_table.shape[0]

def legal_actions(board):
    # Return list of legal moves in algebraic notation
    return [board.san(move) for move in board.legal_moves]

def index_to_action(idx, actions):
    # Map index to action from legal actions
    if 0 <= idx < len(actions):
        return actions[idx]
    return None

# --- GPU Q-learning Training Function ---
def train_qtable_gpu(q_table, episodes=100, gamma=0.9, alpha=0.5, epsilon=0.2, reward_mate=100, reward_draw=-50, reward_step=-0.1):
    for ep in range(episodes):
        fen = State.random_kr_vs_k_fen()
        board = chess.Board(fen)
        done = False
        total_reward = 0
        while not done:
            state_idx = encode_state(board.fen())
            legal_moves = legal_actions(board)
            num_actions = len(legal_moves)
            if num_actions == 0:
                print("No valid actions, skipping.")
                break
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action_idx = random.randint(0, num_actions-1)
            else:
                q_vals = q_table[state_idx][:num_actions]
                action_idx = torch.argmax(q_vals).item()
            action = index_to_action(action_idx, legal_moves)
            if action is None:
                print("No valid action, skipping.")
                break
            board.push_san(action)
            # Reward
            if board.is_game_over():
                done = True
                result = board.result()
                if result == '1-0':
                    reward = reward_mate
                elif result == '0-1':
                    reward = reward_draw
                else:
                    reward = reward_draw
            else:
                reward = reward_step
            total_reward += reward
            # Q-learning update
            next_state_idx = encode_state(board.fen())
            next_legal_moves = legal_actions(board)
            next_num_actions = len(next_legal_moves)
            if next_num_actions == 0:
                max_next_q = 0
            else:
                max_next_q = torch.max(q_table[next_state_idx][:next_num_actions]).item()
            old_q = q_table[state_idx, action_idx].item()
            new_q = old_q + alpha * (reward + gamma * max_next_q - old_q)
            q_table[state_idx, action_idx] = new_q
        print(f"Episode {ep+1}: Total reward: {total_reward}")

# --- Run Training and Save Q-table ---

# --- Q-table Initialization (example shape, adjust as needed) ---
# Example: 10000 states, 50 actions
q_table = torch.zeros((10000, 50), device='cuda')


if __name__ == "__main__":
    # --- Q-table Initialization (example shape, adjust as needed) ---
    # Example: 10000 states, 50 actions
    q_table = torch.zeros((10000, 50), device='cuda')
    train_qtable_gpu(q_table, episodes=100, gamma=0.9, alpha=0.5, epsilon=0.2)
    torch.save(q_table, 'results/qtable_gpu.pt')
    print("Q-table saved to results/qtable_gpu.pt")
        # Epsilon-greedy action selection
        if random.random() < EPSILON:
            action_idx = random.randint(0, num_actions-1)
        else:
            q_vals = q_table[state_idx][:num_actions]
            action_idx = torch.argmax(q_vals).item()
        action = index_to_action(action_idx, legal_moves)
        if action is None:
            print("No valid action, skipping.")
            break
        board.push_san(action)
        # Reward
        if board.is_game_over():
            done = True
            result = board.result()
            reward = REWARD_MATE if result == '1-0' else REWARD_DRAW if result == '0-1' else REWARD_DRAW
        else:
            reward = REWARD_STEP
        total_reward += reward
        # Q-learning update
        next_state_idx = encode_state(board.fen())
        if next_state_idx is None:
            max_next_q = 0
        else:
            next_legal_moves = legal_actions(board)
            next_num_actions = sum([m is not None for m in next_legal_moves])
            if next_num_actions == 0:
                max_next_q = 0
            else:
                max_next_q = torch.max(q_table[next_state_idx][:next_num_actions]).item()
        old_q = q_table[state_idx + (action_idx,)].item()
        new_q = old_q + ALPHA * (reward + GAMMA * max_next_q - old_q)
        q_table[state_idx + (action_idx,)] = new_q
    print(f"Episode {ep+1}: Total reward: {total_reward}")
    print(f"Episode {ep+1}: Total reward: {total_reward}")

# --- Save Q-table ---
torch.save(q_table, 'results/qtable_gpu.pt')
print("Q-table saved to results/qtable_gpu.pt")
