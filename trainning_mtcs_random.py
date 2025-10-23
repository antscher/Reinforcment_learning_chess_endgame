
import math
import random
import time
from typing import List, Tuple, Optional

import chess

from states import State

class MCTSNode:
	def __init__(self, state, parent=None, move=None):
		self.state = state  # chess.Board object
		self.parent = parent
		self.move = move  # move that led to this node
		self.children = []
		self.visits = 0
		self.value = 0.0

	def is_fully_expanded(self):
		legal_moves = list(self.state.legal_moves)
		return len(self.children) == len(legal_moves)

	def best_child(self, c_param=1.4):
		choices_weights = [
			(child.value / (child.visits + 1e-8)) + c_param * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-8))
			for child in self.children
		]
		return self.children[choices_weights.index(max(choices_weights))]

	def expand(self):
		tried_moves = [child.move for child in self.children]
		for move in self.state.legal_moves:
			if move not in tried_moves:
				next_state = self.state.copy()
				next_state.push(move)
				child_node = MCTSNode(next_state, parent=self, move=move)
				self.children.append(child_node)
				return child_node
		return None

	def is_terminal(self):
		return self.state.is_game_over()

	def rollout(self):
		rollout_state = self.state.copy()
		while not rollout_state.is_game_over():
			legal_moves = list(rollout_state.legal_moves)
			move = random.choice(legal_moves)
			rollout_state.push(move)
		result = rollout_state.result()
		if result == '1-0':
			return 1
		elif result == '0-1':
			return -1
		else:
			return 0

	def backpropagate(self, result):
		self.visits += 1
		self.value += result
		if self.parent:
			self.parent.backpropagate(result)


class MCTSTrainer:
	def __init__(self, state_class=None):
		self.state_class = state_class

	def train(self, episodes=100, initial_fen=" ", max_simulations=100):
		start_time = time.time()
		episode_results = []
		for ep in range(episodes):
			board = chess.Board(initial_fen)
			root = MCTSNode(board)
			for _ in range(max_simulations):
				node = root
				# Selection
				while not node.is_terminal() and node.is_fully_expanded():
					node = node.best_child()
				# Expansion
				if not node.is_terminal():
					expanded_node = node.expand()
					if expanded_node is not None:
						node = expanded_node
				# Simulation
				if node is not None:
					result = node.rollout()
					# Backpropagation
					node.backpropagate(result)
			# Choose best move from root
			if root.children:
				best_move = max(root.children, key=lambda c: c.visits).move
				board.push(best_move)
			# Play random moves for black until game ends
			while not board.is_game_over():
				legal_moves = list(board.legal_moves)
				move = random.choice(legal_moves)
				board.push(move)
			result = board.result()
			episode_results.append(result)
		end_time = time.time()
		print(f"MCTS training completed in {end_time - start_time:.2f} seconds.")
		return episode_results, end_time - start_time

	def save_tree(self, root, filename):
		"""
		Save the MCTS tree from the given root node to a file as a list of paths (moves from root to leaf).
		Each path is a sequence of moves leading to a terminal node.
		"""
		def get_paths(node, path, paths):
			if not node.children:
				paths.append([m.uci() for m in path if m is not None])
			else:
				for child in node.children:
					get_paths(child, path + [child.move], paths)
		paths = []
		get_paths(root, [], paths)
		import json
		with open(filename, 'w') as f:
			json.dump(paths, f)
