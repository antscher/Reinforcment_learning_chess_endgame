import random
import time
from typing import List, Tuple, Optional

import chess

from states.QTable import QTable


# ================================
# Q-Learning for KQ vs K Endgame
# ================================
# This script trains a Q-learning agent to checkmate with King+Queen vs King.
# It uses python-chess to ensure legal positions and moves, and stores the
# Q-table with state keys as FEN strings and action keys as UCI moves.
#
# Heavily commented to explain each step and the design rationale.


# ----------------
# Hyperparameters
# ----------------
ALPHA: float = 0.4             # Initial learning rate for Q-value updates
GAMMA: float = 0.95            # Discount factor to value future rewards

# Exploration schedule: start higher, decay slower for broader coverage early on
EPSILON_START: float = 0.6     # Initial exploration rate for epsilon-greedy
EPSILON_END: float = 0.05      # Floor on exploration
EPSILON_DECAY: float = 0.9995  # Per-episode epsilon decay factor

# Rewards: stronger terminal signals, gentler step penalty
STEP_PENALTY: float = -0.005   # Small negative reward per white move to encourage faster mates
MATE_REWARD: float = 5.0       # Strong positive for delivering checkmate
STALEMATE_REWARD: float = -2.0 # Larger penalty for stalemate
FIFTY_MOVE_REWARD: float = -3.0# Penalty for 50-move rule draw

# Shaping: potential-based reward shaping to guide towards good endgame geometry
SHAPING_BETA: float = 0.2      # Weight for potential shaping term
CHECK_BONUS: float = 0.02      # Tiny bonus for giving check without mating (encourages pressure)
MOBILITY_SHAPING_WEIGHT: float = 0.02  # Reward for reducing black king mobility after white move

MAX_WHITE_MOVES_PER_EPISODE: int = 200  # Higher cap to allow completion
SAVE_EVERY_N_EPISODES: int = 100        # Save Q-table periodically

# Learning-rate schedule (simple exponential decay)
ALPHA_END: float = 0.05
ALPHA_DECAY: float = 0.9995

# Curriculum for start positions: begin with BK near the edge more often
USE_CURRICULUM: bool = True
EDGE_BIAS_START: float = 0.9   # probability BK starts near edge at episode 1
EDGE_BIAS_END: float = 0.2     # probability BK starts near edge at last episode

# Black policy during training
BLACK_POLICY: str = "flee"     # options: "random", "flee"

# Periodic evaluation and adaptive opponent switching
EVAL_EVERY: int = 500
EVAL_EPISODES: int = 50
SWITCH_TO_RANDOM_THRESHOLD: float = 0.6  # switch to random BK when win rate >= 60%


# --------------------------
# Position / Action helpers
# --------------------------

def is_valid_kq_vs_k(board: chess.Board) -> bool:
    """
    Validate that the board is a legal KQ vs K position with White to move.

    Conditions:
    - Exactly three pieces: White King, White Queen, Black King
    - Kings are not adjacent (illegal in chess)
    - Side to move is White (agent controls White)
    - Position is legal per python-chess
    """
    # Must be legal
    if not board.is_valid():
        return False

    # White to move so agent controls moves
    if board.turn is not chess.WHITE:
        return False

    # Count pieces
    piece_map = board.piece_map()
    pieces = list(piece_map.values())
    if len(pieces) != 3:
        return False

    # Required piece counts
    num_wk = sum(1 for p in pieces if p.piece_type == chess.KING and p.color == chess.WHITE)
    num_wq = sum(1 for p in pieces if p.piece_type == chess.QUEEN and p.color == chess.WHITE)
    num_bk = sum(1 for p in pieces if p.piece_type == chess.KING and p.color == chess.BLACK)

    if not (num_wk == 1 and num_wq == 1 and num_bk == 1):
        return False

    # Kings not adjacent
    wk_square = next(sq for sq, p in piece_map.items() if p.piece_type == chess.KING and p.color == chess.WHITE)
    bk_square = next(sq for sq, p in piece_map.items() if p.piece_type == chess.KING and p.color == chess.BLACK)
    if chess.square_distance(wk_square, bk_square) <= 1:
        return False

    return True


def random_kq_vs_k_position(edge_bias: float = 0.0) -> chess.Board:
    """
    Generate a random legal KQ vs K position with White to move.

    Method:
    - Randomly place BK, WK, and WQ on distinct squares
    - Reject until position is legal and kings are not adjacent
    - Set side to move White, clear castling/ep metadata appropriately
    """
    while True:
        board = chess.Board.empty()

        # Possibly bias BK towards edge squares to simplify early learning
        if random.random() < edge_bias:
            edge_squares = [sq for sq in chess.SQUARES if _distance_to_edge(sq) == 0]
            bk = random.choice(edge_squares)
        else:
            bk = random.choice(chess.SQUARES)

        wk = random.choice([sq for sq in chess.SQUARES if sq != bk])
        wq = random.choice([sq for sq in chess.SQUARES if sq not in (bk, wk)])

        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(wq, chess.Piece(chess.QUEEN, chess.WHITE))

        # White to move, no castling, reset clocks
        board.turn = chess.WHITE
        board.clear_stack()
        board.halfmove_clock = 0
        board.fullmove_number = 1

        if is_valid_kq_vs_k(board):
            return board


def _distance_to_edge(square: int) -> int:
    """
    King+Queen vs King winning technique generally drives the lone king to the edge/corner.
    This helper returns the Chebyshev distance from a square to the nearest edge (0 on edge).
    """
    file_idx = chess.square_file(square)
    rank_idx = chess.square_rank(square)
    return min(file_idx, 7 - file_idx, rank_idx, 7 - rank_idx)


def _chebyshev_distance(a: int, b: int) -> int:
    """
    Chebyshev distance (king moves metric) between two squares.
    """
    return chess.square_distance(a, b)


def _potential(board: chess.Board) -> float:
    """
    Potential function used for reward shaping. Higher is better for White.

    Intuition:
    - Push the black king towards the board edge (closer to 0 distance from edge is good).
    - Bring the white king closer to the black king for mating net setup.

    We combine these as a weighted sum. The absolute scales are modest, and the shaping is
    applied as a potential-based term to preserve optimal policies in theory.
    """
    piece_map = board.piece_map()

    # Extract square locations
    bk_square = next(sq for sq, p in piece_map.items() if p.piece_type == chess.KING and p.color == chess.BLACK)
    wk_square = next(sq for sq, p in piece_map.items() if p.piece_type == chess.KING and p.color == chess.WHITE)

    # 1) Encourage BK to move towards edge: smaller distance is better
    edge_term = -(float(_distance_to_edge(bk_square)))

    # 2) Encourage WK to approach BK: smaller Chebyshev distance is better
    wk_bk_term = -(float(_chebyshev_distance(wk_square, bk_square)))

    # Weighted sum (weights implicitly 1.0 here; keep simple and stable)
    return edge_term + 0.5 * wk_bk_term


def _black_policy_move(board: chess.Board, legal_moves: List[chess.Move]) -> chess.Move:
    """
    Choose a black move according to the configured policy.
    - random: uniform random
    - flee: prefer moves that increase distance from white king and from edges (towards center)
    """
    if BLACK_POLICY == "random":
        return random.choice(legal_moves)

    # Flee policy
    piece_map = board.piece_map()
    bk_square = next(sq for sq, p in piece_map.items() if p.piece_type == chess.KING and p.color == chess.BLACK)
    wk_square = next(sq for sq, p in piece_map.items() if p.piece_type == chess.KING and p.color == chess.WHITE)

    def flee_score(move: chess.Move) -> float:
        # Simulate BK move destination
        dest = move.to_square
        # Prefer larger distance to white king and larger distance to edge (i.e., center)
        dist_wk = float(_chebyshev_distance(dest, wk_square))
        dist_edge = float(_distance_to_edge(dest))
        return dist_wk + 0.5 * dist_edge

    return max(legal_moves, key=flee_score)


def get_white_actions(board: chess.Board) -> List[str]:
    """
    Return list of legal white actions as UCI strings, restricted to moves
    made by White King or White Queen only.
    """
    actions: List[str] = []
    for move in board.legal_moves:
        piece = board.piece_at(move.from_square)
        if piece is None:
            continue
        if piece.color == chess.WHITE and piece.piece_type in (chess.KING, chess.QUEEN):
            actions.append(move.uci())
    return actions


def apply_white_then_black(board: chess.Board, white_move_uci: str) -> Tuple[chess.Board, float, bool, str]:
    """
    Apply a white move, then (if game not over) a black response.

    Returns: (next_board, reward, done, terminal_reason)
    - reward: immediate reward after black's response (or after white move if terminal)
    - done: whether episode terminated
    - terminal_reason: string for debugging/logging
    """
    terminal_reason = ""

    # Clone board to avoid in-place modification
    next_board = board.copy(stack=False)

    # Potential at current state (for shaping)
    phi_current = _potential(board)

    # 1) Apply white move
    try:
        next_board.push_uci(white_move_uci)
    except Exception:
        # This should not happen if we filtered legal moves, return strong penalty
        return board, -1.0, True, "illegal_white_move"

    # Check terminal after white move
    if next_board.is_checkmate():
        return next_board, MATE_REWARD, True, "checkmate_by_white"
    if next_board.is_stalemate():
        return next_board, STALEMATE_REWARD, True, "stalemate_after_white"
    if next_board.can_claim_fifty_moves() or next_board.halfmove_clock >= 100:
        return next_board, FIFTY_MOVE_REWARD, True, "fifty_move_rule_after_white"

    # Small bonus for giving check without immediately mating (promotes pressure without overfitting)
    check_bonus = CHECK_BONUS if next_board.is_check() else 0.0

    # Mobility shaping: reward states that restrict BK legal moves after white move
    bk_legal_after_white = 0
    for mv in next_board.legal_moves:
        piece = next_board.piece_at(mv.from_square)
        if piece and piece.piece_type == chess.KING and piece.color == chess.BLACK:
            bk_legal_after_white += 1

    # 2) Black random legal reply (only black king exists)
    legal = list(next_board.legal_moves)
    if not legal:
        # Should be covered by stalemate/checkmate, but double-check
        if next_board.is_checkmate():
            return next_board, MATE_REWARD, True, "checkmate_by_white"
        return next_board, STALEMATE_REWARD, True, "stalemate_after_white_no_moves"

    # Choose a black move based on training policy
    black_move = _black_policy_move(next_board, legal)
    next_board.push(black_move)

    # Terminal after black move
    if next_board.is_checkmate():
        # If it's checkmate now, it means black delivered mate which is impossible in KQvK
        return next_board, STALEMATE_REWARD, True, "unexpected_black_mate"
    if next_board.is_stalemate():
        return next_board, STALEMATE_REWARD, True, "stalemate_after_black"
    if next_board.can_claim_fifty_moves() or next_board.halfmove_clock >= 100:
        return next_board, FIFTY_MOVE_REWARD, True, "fifty_move_rule_after_black"

    # Non-terminal step penalty to encourage quicker mating nets + shaping
    phi_next = _potential(next_board)
    # Mobility of BK after black's move (for next state) for consistency we re-count below after black moves
    shaped = STEP_PENALTY + SHAPING_BETA * (GAMMA * phi_next - phi_current) + check_bonus
    return next_board, shaped, False, "in_progress"


# ------------------
# Training routine
# ------------------

def epsilon_greedy_action(qtable: QTable, state_key: str, actions: List[str], epsilon: float) -> Optional[str]:
    """
    Select an action using epsilon-greedy policy.
    - With probability epsilon: choose a random action
    - Otherwise: choose the best known action from Q-table, defaulting to random if unknown
    """
    if not actions:
        return None

    if random.random() < epsilon:
        return random.choice(actions)

    # Exploit: pick argmax from Q-table if present, otherwise fall back to random
    if state_key in qtable.q_table:
        # Ensure unseen actions have a default value in table
        known = qtable.q_table[state_key]
        if known:
            # Choose the action with the highest Q-value among available actions
            best = max(actions, key=lambda a: known.get(a, 0.0))
            return best

    return random.choice(actions)


def ensure_state_actions(qtable: QTable, state_key: str, actions: List[str]) -> None:
    """
    Ensure the Q-table has an entry for this state with all provided actions initialized.
    """
    if state_key not in qtable.q_table:
        qtable.q_table[state_key] = {a: 0.0 for a in actions}
    else:
        # Add any new actions not yet present
        for a in actions:
            if a not in qtable.q_table[state_key]:
                qtable.q_table[state_key][a] = 0.0


def train(num_episodes: int = 2000, seed: Optional[int] = 42, save_path: str = "qtable_kq_vs_k.json") -> QTable:
    """
    Train a Q-learning agent for KQ vs K.

    Args:
    - num_episodes: number of training episodes (each episode is a random starting position)
    - seed: optional RNG seed for reproducibility
    - save_path: where to save the learned Q-table

    Returns:
    - The populated QTable instance
    """
    if seed is not None:
        random.seed(seed)

    qtable = QTable()

    epsilon = EPSILON_START
    alpha = ALPHA

    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        # Curriculum edge bias
        if USE_CURRICULUM and num_episodes > 1:
            t = (episode - 1) / (num_episodes - 1)
            edge_bias = EDGE_BIAS_START * (1.0 - t) + EDGE_BIAS_END * t
        else:
            edge_bias = 0.0

        # New random starting position (possibly edge-biased)
        board = random_kq_vs_k_position(edge_bias=edge_bias)

        white_moves_this_episode = 0

        # Episode loop (white moves only counted)
        while white_moves_this_episode < MAX_WHITE_MOVES_PER_EPISODE:
            state_key = board.fen()
            actions = get_white_actions(board)

            # If no actions, it's terminal or illegal state
            if not actions:
                break

            # Make sure Q-table has this state/actions initialized
            ensure_state_actions(qtable, state_key, actions)

            # Choose action via epsilon-greedy
            action = epsilon_greedy_action(qtable, state_key, actions, epsilon)
            if action is None:
                break

            # Apply action: white then black
            next_board, reward, done, reason = apply_white_then_black(board, action)

            # Prepare next state's info for update
            next_state_key = next_board.fen()
            next_actions = get_white_actions(next_board) if not done else []

            # Q-learning update
            old_q = qtable.q_table[state_key].get(action, 0.0)
            if done or not next_actions:
                target = reward
            else:
                # Ensure next state's actions exist before taking max
                ensure_state_actions(qtable, next_state_key, next_actions)
                next_max = max(qtable.q_table[next_state_key].get(a, 0.0) for a in next_actions)
                target = reward + GAMMA * next_max

            # Decayed learning rate
            new_q = (1 - alpha) * old_q + alpha * target
            qtable.q_table[state_key][action] = new_q

            # Move forward in environment
            board = next_board
            white_moves_this_episode += 1

            if done:
                break

        # Epsilon and alpha decay per episode (bounded)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        alpha = max(ALPHA_END, alpha * ALPHA_DECAY)

        # Periodic saves
        if episode % SAVE_EVERY_N_EPISODES == 0:
            qtable.save(save_path)

        # Periodic evaluation and adaptive opponent switching
        if episode % 500 == 0:
            wins, stalemates, fifty, avg_moves = evaluate(qtable, episodes=50)
            win_rate = wins / max(1, (wins + stalemates + fifty))
            print(f"Interim eval @ {episode}: win_rate={win_rate:.2f}, avg_moves={avg_moves:.1f}")
            global BLACK_POLICY
            if BLACK_POLICY == "flee" and win_rate >= 0.6:
                BLACK_POLICY = "random"
                print("Switching BLACK_POLICY to random for generalization.")

        # Console progress (every 50 episodes)
        if episode % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Episode {episode}/{num_episodes} | eps={epsilon:.3f} | alpha={alpha:.3f} | edge_bias={edge_bias:.2f} | elapsed={elapsed:.1f}s")

    # Final save
    qtable.save(save_path)
    total_elapsed = time.time() - start_time
    print(f"Training complete. Episodes: {num_episodes}, time: {total_elapsed:.1f}s, saved: {save_path}")

    return qtable


# ------------------
# Evaluation helper
# ------------------

def evaluate(qtable: QTable, episodes: int = 100, seed: Optional[int] = 123) -> Tuple[int, int, int, float]:
    """
    Evaluate the greedy policy derived from Q-table over random starting positions.

    Returns: (wins, stalemates, fifty_move_draws, avg_white_moves_per_game)
    """
    if seed is not None:
        random.seed(seed)

    wins = 0
    stalemates = 0
    fifty_draws = 0
    moves_sum = 0

    for _ in range(episodes):
        board = random_kq_vs_k_position()
        white_moves = 0

        for _ in range(MAX_WHITE_MOVES_PER_EPISODE):
            state_key = board.fen()
            actions = get_white_actions(board)
            if not actions:
                break

            # Greedy choice from learned values; default to random if unseen
            if state_key in qtable.q_table and qtable.q_table[state_key]:
                best_action = max(actions, key=lambda a: qtable.q_table[state_key].get(a, 0.0))
            else:
                best_action = random.choice(actions)

            board, reward, done, reason = apply_white_then_black(board, best_action)
            white_moves += 1

            if done:
                if reason.startswith("checkmate"):
                    wins += 1
                elif "stalemate" in reason:
                    stalemates += 1
                elif "fifty_move" in reason:
                    fifty_draws += 1
                break

        moves_sum += white_moves

    avg_moves = moves_sum / max(1, episodes)
    print(f"Evaluation: wins={wins}, stalemates={stalemates}, fifty_draws={fifty_draws}, avg_white_moves={avg_moves:.1f}")
    return wins, stalemates, fifty_draws, avg_moves


if __name__ == "__main__":
    # Example usage:
    # 1) Train for a number of episodes
    q = train(num_episodes=2000, save_path="qtable_kq_vs_k.json")

    # 2) Evaluate the learned policy
    evaluate(q, episodes=100)
