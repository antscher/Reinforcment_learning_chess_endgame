class QTable:
    def argmax_epsilon(self, state, epsilon):
        """
        Epsilon-greedy action selection: with probability epsilon, choose a random action;
        otherwise, choose the action with the highest value.
        Args:
            state: Hashable representation of the state
            epsilon: Probability of choosing a random action (float between 0 and 1)
        Returns:
            Selected action or None if state not found
        """
        import random
        if state not in self.q_table or not self.q_table[state]:
            return None
        actions = list(self.q_table[state].keys())
        if random.random() < epsilon:
            return random.choice(actions)
        return max(self.q_table[state], key=self.q_table[state].get)
    
    def save(self, filename):
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
        with open(filename, 'w') as f:
            json.dump(serializable_qtable, f)
            
    def __init__(self):
        # Dictionary mapping state to action-value dictionary
        self.q_table = {}

    def add_state(self, state, actions):
        """
        Add a state to the Q-table with actions initialized to 0.
        Args:
            state: Hashable representation of the state
            actions: List of possible actions for the state (UCI format)
        """
        # Transform UCI actions to algebraic notation using State method
        if state not in self.q_table:
            transformed_actions = [state.uci_to_algebraic(action) for action in actions]
            self.q_table[state] = {action: 0.0 for action in transformed_actions}

    def argmax(self, state):
        """
        Return the action with the highest value for the given state.
        Args:
            state: Hashable representation of the state
        Returns:
            The action with the maximum Q-value, or None if state not found
        """
        if state not in self.q_table or not self.q_table[state]:
            return None
        return max(self.q_table[state], key=self.q_table[state].get)

    def print_state(self, state):
        """
        Print the action-value dictionary for a given state.
        Args:
            state: Hashable representation of the state
        """
        if state in self.q_table:
            print(self.q_table[state])
        else:
            print(f"State {state} not found in Q-table.")

    def set_action_value(self, state, action, value):
        """
        Set the value of a specific action for a given state.
        Args:
            state: Hashable representation of the state
            action: The action to update
            value: The new value to set
        """
        if state in self.q_table and action in self.q_table[state]:
            self.q_table[state][action] = value
        else:
            raise KeyError(f"State {state} or action {action} not found in Q-table.")
