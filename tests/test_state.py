
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from states.state_KRvsk import State
from states.QTable import QTable

# Test symmetry/rotation generation for FEN and action
print("\nSymmetries and rotations for FEN and action:")
sample_fen = "8/8/8/8/8/2R5/k1K5/8"
sample_action = "Ra3"
symmetries = State.fen_action_symmetries(sample_fen, sample_action)
for i, (fen, action) in enumerate(symmetries):
    print(f"{i+1}: FEN: {fen}, Action: {action}")

# Test random KR vs k FEN generation
print("Random KR vs k FEN:")
random_fen = State.random_kr_vs_k_fen()
print(random_fen)
random_state = State()
random_state.create_from_fen(random_fen)
random_state.print_state()


# Test State class
fen = "8/8/1k6/8/8/4K3/3R4/8 w - - 0 1"
state = State()
# Create state from FEN
state.create_from_fen(fen)
# Print state pieces
state.print_state()

new_state = state.get_new_state("Rd6")

# Print new state pieces
print("New state after move Rd6:")
new_state.print_state()


# Test QTable class
qtable = QTable()
fen_key = fen
# some possible ACTIONS in UCI format
uci_actions = ["Rd6", "Rd7", "Rd8"]
# Add state and actions to QTable
qtable.add_state(fen_key, uci_actions)
qtable.print_state(fen_key)
# Set value for an action (algebraic notation)
qtable.set_action_value(fen_key, "Rd7", 1.5)
# Print action-value dictionary for the state
qtable.print_state(fen_key)
# Get the action with the highest value
best_action = qtable.argmax(fen_key)
print(f"Best action for the state: {best_action}")
# Save Q-table to JSON file
qtable.save("results","qtable_test.json")