from states.state_KRvsk import State
from states.QTable import QTable

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
state_key = state
#some possible ACTIONS
actions = ["Rd6", "Rd7", "Rd8"]
# Add state and actions to QTable
qtable.add_state(state_key, actions)
qtable.print_state(state_key)
# Set value for an action
qtable.set_action_value(state_key, "Rd7", 1.5)
# Print action-value dictionary for the state
qtable.print_state(state_key)
# Get the action with the highest value
best_action = qtable.argmax(state_key)
print(f"Best action for the state: {best_action}")
# Save Q-table to JSON file
qtable.save("qtable_test.json")