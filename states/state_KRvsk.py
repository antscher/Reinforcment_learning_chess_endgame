import chess

class State:
    @staticmethod
    def fen_uci_symmetries(fen, uci_move):
        """
        Given a FEN and a UCI move, return all FENs and UCI moves associated with 1 or 2 symmetries/rotations using State class.
        Returns:
            List of (fen, uci_move) tuples
        """
        def rotate_square(square, rot):
            r, c = square
            if rot == 0:
                return r, c
            elif rot == 90:
                return c, 7 - r
            elif rot == 180:
                return 7 - r, 7 - c
            elif rot == 270:
                return 7 - c, r
        def mirror_square(square, axis):
            r, c = square
            if axis == 'h':
                return 7 - r, c
            elif axis == 'v':
                return r, 7 - c
        def transform_uci_move(uci_move, transform_fn):
            from_sq = uci_move[:2]
            to_sq = uci_move[2:4]
            col_from = ord(from_sq[0]) - ord('a')
            row_from = 8 - int(from_sq[1])
            col_to = ord(to_sq[0]) - ord('a')
            row_to = 8 - int(to_sq[1])
            new_row_from, new_col_from = transform_fn((row_from, col_from))
            new_row_to, new_col_to = transform_fn((row_to, col_to))
            new_from_sq = chr(new_col_from + ord('a')) + str(8 - new_row_from)
            new_to_sq = chr(new_col_to + ord('a')) + str(8 - new_row_to)
            return new_from_sq + new_to_sq + uci_move[4:]  # preserve promotion if present

        state = State()
        state.create_from_fen(fen)
        piece_map = {
            'K': state.pieces['WKing'],
            'R': state.pieces['WRook'],
            'k': state.pieces['BKing']
        }

        results = set()
        transforms = [
            lambda sq: rotate_square(sq, 0),
            lambda sq: rotate_square(sq, 90),
            lambda sq: rotate_square(sq, 180),
            lambda sq: rotate_square(sq, 270),
            lambda sq: mirror_square(sq, 'h'),
            lambda sq: mirror_square(sq, 'v'),
        ]
        for i, tf1 in enumerate(transforms):
            new_map = {p: tf1(pos) if pos is not None else None for p, pos in piece_map.items()}
            if any(pos is not None and (not isinstance(pos, tuple) or len(pos) != 2) for pos in new_map.values()):
                continue
            new_state = State(
                w_king=new_map['K'],
                w_rook=new_map['R'],
                b_king=new_map['k']
            )
            new_fen = new_state.to_fen().split(' ')[0]
            new_uci = None
            if uci_move is not None:
                new_uci = transform_uci_move(uci_move, tf1)
            results.add((new_fen, new_uci))
            for j, tf2 in enumerate(transforms):
                if i == j:
                    continue
                new_map2 = {p: tf2(tf1(pos)) if pos is not None else None for p, pos in piece_map.items()}
                if any(pos is not None and (not isinstance(pos, tuple) or len(pos) != 2) for pos in new_map2.values()):
                    continue
                new_state2 = State(
                    w_king=new_map2['K'],
                    w_rook=new_map2['R'],
                    b_king=new_map2['k']
                )
                new_fen2 = new_state2.to_fen().split(' ')[0]
                new_uci2 = None
                if uci_move is not None:
                    new_uci2 = transform_uci_move(uci_move, lambda sq: tf2(tf1(sq)))
                results.add((new_fen2, new_uci2))
        return list(results)
    @staticmethod
    def random_kr_vs_k_fen_column_a():
        """
        Generate a random valid FEN for KR vs k endgame with:
        - Black king in column 'a' (col=0), row between 1 and 4
        - White king in same row, column 'c' (col=2)
        - Rook anywhere else except those squares
        - Ensures kings are not adjacent and black is not in check
        Returns:
            str: Valid FEN string
        """
        import random
        import chess

        def are_kings_adjacent(wk, bk):
            return max(abs(wk[0] - bk[0]), abs(wk[1] - bk[1])) <= 1

        def is_between(a, b, c):
            return min(a, b) < c < max(a, b)

        while True:
            bk_row = random.choice([1, 2, 3, 4])
            bk_col = 0
            wk_row = bk_row
            wk_col = 2
            # Rook can be anywhere except (bk_row, bk_col) and (wk_row, wk_col)
            rook_positions = [(r, c) for r in range(8) for c in range(8)
                             if (r, c) != (bk_row, bk_col) and (r, c) != (wk_row, wk_col)]
            wr_row, wr_col = random.choice(rook_positions)

            wk = (wk_row, wk_col)
            wr = (wr_row, wr_col)
            bk = (bk_row, bk_col)

            if are_kings_adjacent(wk, bk):
                continue

            state = State(w_king=wk, w_rook=wr, b_king=bk)
            fen = state.to_fen()

            # Custom check: rook attacks black king in row/col, unless white king is between
            in_check = False
            # Rook and black king in same row
            if wr_row == bk_row:
                # Is white king between rook and black king in same row?
                if wk_row == wr_row and is_between(wr_col, bk_col, wk_col):
                    pass
                else:
                    in_check = True
            # Rook and black king in same column
            if wr_col == bk_col:
                # Is white king between rook and black king in same column?
                if wk_col == wr_col and is_between(wr_row, bk_row, wk_row):
                    pass
                else:
                    in_check = True
            if in_check:
                continue
            return fen
    @staticmethod
    def fen_action_symmetries(fen, action):
        """
        Given a FEN and an action, return all FENs and actions associated with 1 or 2 symmetries/rotations using State class.
        Returns:
            List of (fen, action) tuples
        """
        def rotate_square(square, rot):
            r, c = square
            if rot == 0:
                return r, c
            elif rot == 90:
                return c, 7 - r
            elif rot == 180:
                return 7 - r, 7 - c
            elif rot == 270:
                return 7 - c, r
        def mirror_square(square, axis):
            r, c = square
            if axis == 'h':
                return 7 - r, c
            elif axis == 'v':
                return r, 7 - c
        def transform_action(action, transform_fn):
            piece = action[0]
            col = ord(action[1]) - ord('a')
            row = 8 - int(action[2])
            new_row, new_col = transform_fn((row, col))
            new_square = chr(new_col + ord('a')) + str(8 - new_row)
            return piece + new_square

        # Use State to parse FEN
        state = State()
        state.create_from_fen(fen)
        piece_map = {
            'K': state.pieces['WKing'],
            'R': state.pieces['WRook'],
            'k': state.pieces['BKing']
        }

        results = set()
        transforms = [
            lambda sq: rotate_square(sq, 0),
            lambda sq: rotate_square(sq, 90),
            lambda sq: rotate_square(sq, 180),
            lambda sq: rotate_square(sq, 270),
            lambda sq: mirror_square(sq, 'h'),
            lambda sq: mirror_square(sq, 'v'),
        ]
        for i, tf1 in enumerate(transforms):
            # Apply one transformation
            new_map = {p: tf1(pos) if pos is not None else None for p, pos in piece_map.items()}
            # Ensure all positions are valid tuples
            if any(pos is not None and (not isinstance(pos, tuple) or len(pos) != 2) for pos in new_map.values()):
                continue
            new_state = State(
                w_king=new_map['K'],
                w_rook=new_map['R'],
                b_king=new_map['k']
            )
            new_fen = new_state.to_fen().split(' ')[0]
            new_action = None
            if action is not None:
                new_action = transform_action(action, tf1)
            results.add((new_fen, new_action))
            # Apply two transformations
            for j, tf2 in enumerate(transforms):
                if i == j:
                    continue
                new_map2 = {p: tf2(tf1(pos)) if pos is not None else None for p, pos in piece_map.items()}
                if any(pos is not None and (not isinstance(pos, tuple) or len(pos) != 2) for pos in new_map2.values()):
                    continue
                new_state2 = State(
                    w_king=new_map2['K'],
                    w_rook=new_map2['R'],
                    b_king=new_map2['k']
                )
                new_fen2 = new_state2.to_fen().split(' ')[0]
                new_action2 = None
                if action is not None:
                    new_action2 = transform_action(action, lambda sq: tf2(tf1(sq)))
                results.add((new_fen2, new_action2))
        return list(results)
    @staticmethod
    def random_kr_vs_k_fen():
        """
        Generate a random valid FEN for KR vs k endgame using State class.
        Ensures kings are not adjacent and black is not in check.
        Returns:
            str: Valid FEN string
        """
        import random
        import chess

        def are_kings_adjacent(wk, bk):
            return max(abs(wk[0] - bk[0]), abs(wk[1] - bk[1])) <= 1

        while True:
            positions = random.sample([(r, c) for r in range(8) for c in range(8)], 3)
            wk, wr, bk = positions
            if are_kings_adjacent(wk, bk):
                continue
            # Use State to build and convert to FEN
            state = State(w_king=wk, w_rook=wr, b_king=bk)
            fen = state.to_fen()
            # Custom check: rook attacks black king in row/col, unless white king is between
            wk_row, wk_col = wk
            wr_row, wr_col = wr
            bk_row, bk_col = bk

            def is_between(a, b, c):
                return min(a, b) < c < max(a, b)

            in_check = False
            # Rook and black king in same row
            if wr_row == bk_row:
                # Is white king between rook and black king in same row?
                if wk_row == wr_row and is_between(wr_col, bk_col, wk_col):
                    pass
                else:
                    in_check = True
            # Rook and black king in same column
            if wr_col == bk_col:
                # Is white king between rook and black king in same column?
                if wk_col == wr_col and is_between(wr_row, bk_row, wk_row):
                    pass
                else:
                    in_check = True
            if in_check:
                continue
            return fen
        
    """
    A class to represent a chess endgame state with WKing, WRook, and BKing positions.
    Positions are stored as tuples of integers (row, col) where 0 <= row, col <= 7.
    """
    
    def __init__(self, w_king=None, w_rook=None, b_king=None):
        """
        Initialize a chess state.
        
        Args:
            W_king (tuple): Position of W King as (row, col)
            w_rook (tuple): Position of W Rook as (row, col)
            b_king (tuple): Position of B King as (row, col)
        """
        self.pieces = {
            'WKing': w_king,
            'WRook': w_rook,
            'BKing': b_king
        }
    
    def create(self, w_king, w_rook, b_king):
        """
        Create a new state with given positions.
        
        Args:
            w_king (tuple): Position of White King as (row, col)
            w_rook (tuple): Position of White Rook as (row, col)
            b_king (tuple): Position of Black King as (row, col)
        """
        self.pieces['WKing'] = w_king
        self.pieces['WRook'] = w_rook
        self.pieces['BKing'] = b_king

    def create_from_fen(self, fen):
        """
        Create a new state from a FEN string.
        
        Args:
            fen (str): FEN string representing the board state
        """
        piece_map = {
            'K': 'WKing',
            'R': 'WRook',
            'k': 'BKing'
        }
        
        rows = fen.split(' ')[0].split('/')
        for r, row in enumerate(rows):
            c = 0
            for char in row:
                if char.isdigit():
                    c += int(char)
                elif char in piece_map:
                    self.pieces[piece_map[char]] = (r, c)
                    c += 1
                else:
                    c += 1
    
    def update(self, piece_name, new_position):
        """
        Update the position of a specific piece.
        
        Args:
            piece_name (str): Name of the piece ('WKing', 'WRook', 'BKing')
            new_position (tuple): New position as (row, col)
        """
        if piece_name in self.pieces:
            self.pieces[piece_name] = new_position
        else:
            raise ValueError(f"Invalid piece name: {piece_name}")
    
    def get_position(self, piece_name):
        """
        Get the position of a specific piece.
        
        Args:
            piece_name (str): Name of the piece ('WKing', 'WRook', 'BKing')

        Returns:
            tuple: Position as (row, col) or None if piece not set
        """
        return self.pieces.get(piece_name)
    

    def get_new_state(self, uci_move):
        """
        Get a new state resulting from a UCI move.
        
        Args:
            uci_move (str): The UCI move string (e.g., 'Rf3')

        Returns:
            State: A new state reflecting the move
        """
        piece_map = {
            'K': 'WKing',
            'R': 'WRook',
            'k': 'BKing'
        }
        
        piece_char = uci_move[0]
        target_square = uci_move[1:3]
        
        if piece_char not in piece_map:
            raise ValueError(f"Invalid piece character in move: {piece_char}")
        
        piece_name = piece_map[piece_char]
        col = ord(target_square[0]) - ord('a')
        row = 8 - int(target_square[1])
        
        new_state = State(
            w_king=self.pieces['WKing'],
            w_rook=self.pieces['WRook'],
            b_king=self.pieces['BKing']
        )
        new_state.update(piece_name, (row, col))
        
        return new_state
    
    def print_state(self):
        """
        Print the current state of the board.
        """
        board = [['.' for _ in range(8)] for _ in range(8)]
        
        for piece, position in self.pieces.items():
            if position:
                row, col = position
                symbol = 'K' if piece == 'WKing' else 'k' if piece == 'BKing' else 'R'
                board[row][col] = symbol
        
        for row in board:
            print(' '.join(row))

    def to_fen(self):
        """
        Convert the current state to a FEN string.
        
        Returns:
            str: FEN representation of the current state
        """
        board = [['1' for _ in range(8)] for _ in range(8)]
        
        for piece, position in self.pieces.items():
            if position:
                row, col = position
                symbol = 'K' if piece == 'WKing' else 'k' if piece == 'BKing' else 'R'
                board[row][col] = symbol
        
        fen_rows = []
        for row in board:
            fen_row = ''
            empty_count = 0
            for cell in row:
                if cell == '1':
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    fen_row += cell
            if empty_count > 0:
                fen_row += str(empty_count)
            fen_rows.append(fen_row)
        
        fen = '/'.join(fen_rows) + ' w - - 0 1'
        return fen
    
    def uci_to_algebraic(self, uci_move):
        """
        Convert a UCI move (e.g. 'd2d4') to algebraic notation (e.g. 'Kd4', 'Rd4', etc.) using only the state.
        Args:
            uci_move (str): UCI move string
        Returns:
            str: Move in algebraic notation
        """
        from_square = uci_move[:2]
        to_square = uci_move[2:4]
        # Find which piece is at from_square
        for name, pos in self.pieces.items():
            if pos:
                col = chr(pos[1] + ord('a'))
                row = str(8 - pos[0])
                if from_square == f"{col}{row}":
                    if name == 'WKing':
                        symbol = 'K'
                    elif name == 'WRook':
                        symbol = 'R'
                    elif name == 'BKing':
                        symbol = 'k'
                    else:
                        symbol = name[0]
                    return f"{symbol}{to_square}"
        return uci_move
    


