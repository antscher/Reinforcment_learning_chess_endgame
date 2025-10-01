from states.state_KRvsk import State
import os

class QTable:
    def argmax_epsilon(self, fen, epsilon):
        """
        Epsilon-greedy action selection: with probability epsilon, choose a random action;
        otherwise, choose the action with the highest value.
        Args:
            fen: FEN string representing the state
            epsilon: Probability of choosing a random action (float between 0 and 1)
        Returns:
            Selected action or None if state not found
        """
        import random
        if fen not in self.q_table or not self.q_table[fen]:
            return None
        actions = list(self.q_table[fen].keys())
        if random.random() < epsilon:
            return random.choice(actions)
        return max(self.q_table[fen], key=self.q_table[fen].get)

    def save(self, folder, filename):
        """
        Save the Q-table to a file in JSON format.
        Args:
            filename (str): Path to the file where the Q-table will be saved
        """
        import json
        # Convert State objects to FEN string for serialization
        serializable_qtable = {}
        for state, actions in self.q_table.items():
            if hasattr(state, "to_fen"):
                key = state.to_fen()
            else:
                key = str(state)
            serializable_qtable[key] = actions
        with open(os.path.join(folder, filename), 'w') as f:
            json.dump(serializable_qtable, f)
            
    def __init__(self):
        # Dictionary mapping state to action-value dictionary
        self.q_table = {}

    def add_state(self, fen,  actions):
        """
        Add a state to the Q-table with actions initialized to 0.
        Args:
            fen: FEN string representing the state
            state_obj: State object for algebraic conversion
            actions: List of possible actions for the state (UCI format)
        """
        # Transform UCI actions to algebraic notation using State method
        if fen not in self.q_table:
            state = State()
            state.create_from_fen(fen)
            transformed_actions = [state.uci_to_algebraic(action) for action in actions]
            self.q_table[fen] = {action: 0.0 for action in transformed_actions}

    def argmax(self, fen):
        """
        Return the action with the highest value for the given state.
        Args:
            fen: FEN string representing the state
        Returns:
            The action with the maximum Q-value, or None if state not found
        """
        if fen not in self.q_table or not self.q_table[fen]:
            return None
        return max(self.q_table[fen], key=self.q_table[fen].get)

    def print_state(self, fen):
        """
        Print the action-value dictionary for a given state.
        Args:
            fen: FEN string representing the state
        """
        if fen in self.q_table:
            print(self.q_table[fen])
        else:
            print(f"State {fen} not found in Q-table.")

    def set_action_value(self, fen, action, value):
        """
        Set the value of a specific action for a given state.
        Args:
            fen: FEN string representing the state
            action: The action to update
            value: The new value to set
        """
        if fen in self.q_table and action in self.q_table[fen]:
            self.q_table[fen][action] = value
        else:
            raise KeyError(f"State {fen} or action {action} not found in Q-table.")
