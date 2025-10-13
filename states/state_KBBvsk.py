import chess

class State:
    @staticmethod
    def random_kbb_vs_k_fen_column_a():
        """
        Generate a random valid FEN for KBB vs k endgame with:
        - Black king in column 'a' (col=0), row between 1 and 4
        - White king in same row, column 'c' (col=2)
        - Two bishops on different colored squares elsewhere
        - Ensures kings are not adjacent and black is not in check
        Returns:
            str: Valid FEN string
        """
        import random
        import chess

        def are_kings_adjacent(wk, bk):
            return max(abs(wk[0] - bk[0]), abs(wk[1] - bk[1])) <= 1
            
        def is_square_color(row, col, color):
            """Determine if a square is of the specified color (0=dark, 1=light)"""
            return (row + col) % 2 == color
            
        def is_bishop_attacking(bishop_pos, king_pos):
            """Check if a bishop at bishop_pos is attacking king at king_pos"""
            # Bishop attacks diagonally, so the absolute difference in row and column should be equal
            row_diff = abs(bishop_pos[0] - king_pos[0])
            col_diff = abs(bishop_pos[1] - king_pos[1])
            if row_diff == col_diff:
                # Now we need to check if the path is clear
                dr = 1 if bishop_pos[0] < king_pos[0] else -1
                dc = 1 if bishop_pos[1] < king_pos[1] else -1
                r, c = bishop_pos[0] + dr, bishop_pos[1] + dc
                while (r, c) != king_pos:
                    # If there's a piece in the way, the bishop is not attacking
                    if (r, c) == wk_pos:  # Only white king could be in the way
                        return False
                    r += dr
                    c += dc
                return True
            return False

        while True:
            bk_row = random.choice([1, 2, 3, 4])
            bk_col = 0  # Column a
            wk_row = bk_row
            wk_col = 2  # Column c
            
            wk_pos = (wk_row, wk_col)
            bk_pos = (bk_row, bk_col)
            
            # Ensure kings are not adjacent
            if are_kings_adjacent(wk_pos, bk_pos):
                continue
            
            # Generate positions for bishops on different colored squares
            # Exclude squares occupied by kings
            occupied_squares = [wk_pos, bk_pos]
            
            # Get available light and dark squares
            light_squares = [(r, c) for r in range(8) for c in range(8) 
                            if is_square_color(r, c, 1) and (r, c) not in occupied_squares]
            dark_squares = [(r, c) for r in range(8) for c in range(8) 
                            if is_square_color(r, c, 0) and (r, c) not in occupied_squares]
                            
            # Place bishops
            wb1_pos = random.choice(light_squares)
            occupied_squares.append(wb1_pos)
            
            dark_squares = [sq for sq in dark_squares if sq not in occupied_squares]
            wb2_pos = random.choice(dark_squares)
            
            # Manually check if the black king is in check by either bishop
            in_check = is_bishop_attacking(wb1_pos, bk_pos) or is_bishop_attacking(wb2_pos, bk_pos)
            if in_check:
                continue
            
            # Create state and double-check using chess library
            state = State(w_king=wk_pos, w_bishop1=wb1_pos, w_bishop2=wb2_pos, b_king=bk_pos)
            fen = state.to_fen()
            
            board = chess.Board(fen)
            if board.is_check():
                continue  # Retry if black king is in check
                
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
            'B1': state.pieces['WBishop1'],
            'B2': state.pieces['WBishop2'],
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
            new_map = {p: tf1(pos) if pos else None for p, pos in piece_map.items()}
            new_state = State(
                w_king=new_map['K'],
                w_bishop1=new_map['B1'],
                w_bishop2=new_map['B2'],
                b_king=new_map['k']
            )
            new_fen = new_state.to_fen().split(' ')[0]
            new_action = None
            if action is not None:
                new_action = transform_action(action, tf1)
            results.add((new_fen, new_action))
            
            # Apply two transformations
            for j, tf2 in enumerate(transforms):
                if i == j: continue
                new_map2 = {p: tf2(tf1(pos)) if pos else None for p, pos in piece_map.items()}
                new_state2 = State(
                    w_king=new_map2['K'],
                    w_bishop1=new_map2['B1'],
                    w_bishop2=new_map2['B2'],
                    b_king=new_map2['k']
                )
                new_fen2 = new_state2.to_fen().split(' ')[0]
                new_action2 = None
                if action is not None:
                    new_action2 = transform_action(action, lambda sq: tf2(tf1(sq)))
                results.add((new_fen2, new_action2))
                
        return list(results)

    @staticmethod
    def random_kbb_vs_k_fen():
        """
        Generate a random valid FEN for KBB vs k endgame using State class.
        Ensures kings are not adjacent, black is not in check, and bishops are on different colored squares.
        Returns:
            str: Valid FEN string
        """
        import random
        import chess

        def are_kings_adjacent(wk, bk):
            return max(abs(wk[0] - bk[0]), abs(wk[1] - bk[1])) <= 1
            
        def is_square_color(row, col, color):
            """Determine if a square is of the specified color (0=dark, 1=light)"""
            return (row + col) % 2 == color
            
        def is_bishop_attacking(bishop_pos, king_pos, occupied_squares):
            """
            Check if a bishop at bishop_pos is attacking king at king_pos,
            considering occupied_squares as obstacles
            """
            if not bishop_pos or not king_pos:
                return False
                
            # Bishop attacks diagonally, so the absolute difference in row and column should be equal
            row_diff = abs(bishop_pos[0] - king_pos[0])
            col_diff = abs(bishop_pos[1] - king_pos[1])
            
            if row_diff != col_diff:
                return False  # Not on the same diagonal
                
            # Now we need to check if the path is clear
            dr = 1 if bishop_pos[0] < king_pos[0] else -1
            dc = 1 if bishop_pos[1] < king_pos[1] else -1
            r, c = bishop_pos[0] + dr, bishop_pos[1] + dc
            while (r, c) != king_pos:
                # If there's a piece in the way, the bishop is not attacking
                if (r, c) in occupied_squares:
                    return False
                r += dr
                c += dc
            return True

        while True:
            # Place kings first to ensure they're not adjacent
            all_squares = [(r, c) for r in range(8) for c in range(8)]
            wk_pos = random.choice(all_squares)
            remaining_squares = [pos for pos in all_squares if pos != wk_pos]
            
            # Place black king not adjacent to white king
            possible_bk_pos = [pos for pos in remaining_squares if not are_kings_adjacent(wk_pos, pos)]
            if not possible_bk_pos:
                continue  # Retry if no valid position for black king
                
            bk_pos = random.choice(possible_bk_pos)
            occupied_squares = [wk_pos, bk_pos]
            remaining_squares = [pos for pos in remaining_squares if pos != bk_pos]
            
            # Place bishops on different colored squares
            light_squares = [pos for pos in remaining_squares if is_square_color(pos[0], pos[1], 1)]
            dark_squares = [pos for pos in remaining_squares if is_square_color(pos[0], pos[1], 0)]
            
            if not light_squares or not dark_squares:
                continue  # Retry if can't place bishops on different colored squares
            
            # Try several bishop placements to find one where black king is not in check
            valid_position_found = False
            for _ in range(10):  # Try up to 10 different bishop placements
                wb1_pos = random.choice(light_squares)
                wb2_pos = random.choice(dark_squares)
                
                # Manually check if the black king is in check by either bishop
                if (is_bishop_attacking(wb1_pos, bk_pos, [wk_pos]) or 
                    is_bishop_attacking(wb2_pos, bk_pos, [wk_pos])):
                    continue  # Bishop is checking the king, try another placement
                    
                # Create state and double-check using chess library
                state = State(w_king=wk_pos, w_bishop1=wb1_pos, w_bishop2=wb2_pos, b_king=bk_pos)
                fen = state.to_fen()
                
                board = chess.Board(fen)
                if not board.is_check():
                    valid_position_found = True
                    break
            
            if valid_position_found:
                return fen # type: ignore
        
    """
    A class to represent a chess endgame state with WKing, two WBishops, and BKing positions.
    Positions are stored as tuples of integers (row, col) where 0 <= row, col <= 7.
    """
    
    def __init__(self, w_king=None, w_bishop1=None, w_bishop2=None, b_king=None):
        """
        Initialize a chess state.
        
        Args:
            w_king (tuple): Position of White King as (row, col)
            w_bishop1 (tuple): Position of first White Bishop as (row, col)
            w_bishop2 (tuple): Position of second White Bishop as (row, col)
            b_king (tuple): Position of Black King as (row, col)
        """
        self.pieces = {
            'WKing': w_king,
            'WBishop1': w_bishop1,
            'WBishop2': w_bishop2,
            'BKing': b_king
        }
    
    def create(self, w_king, w_bishop1, w_bishop2, b_king):
        """
        Create a new state with given positions.
        
        Args:
            w_king (tuple): Position of White King as (row, col)
            w_bishop1 (tuple): Position of first White Bishop as (row, col)
            w_bishop2 (tuple): Position of second White Bishop as (row, col)
            b_king (tuple): Position of Black King as (row, col)
        """
        self.pieces['WKing'] = w_king
        self.pieces['WBishop1'] = w_bishop1
        self.pieces['WBishop2'] = w_bishop2
        self.pieces['BKing'] = b_king

    def create_from_fen(self, fen):
        """
        Create a new state from a FEN string.
        
        Args:
            fen (str): FEN string representing the board state
        """
        piece_map = {
            'K': 'WKing',
            'B': ['WBishop1', 'WBishop2'],
            'k': 'BKing'
        }
        
        rows = fen.split(' ')[0].split('/')
        bishop_count = 0
        
        for r, row in enumerate(rows):
            c = 0
            for char in row:
                if char.isdigit():
                    c += int(char)
                elif char in piece_map:
                    if char == 'B':
                        self.pieces[piece_map[char][bishop_count]] = (r, c)
                        bishop_count += 1
                    else:
                        self.pieces[piece_map[char]] = (r, c)
                    c += 1
                else:
                    c += 1
    
    def update(self, piece_name, new_position):
        """
        Update the position of a specific piece.
        
        Args:
            piece_name (str): Name of the piece ('WKing', 'WBishop1', 'WBishop2', 'BKing')
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
            piece_name (str): Name of the piece ('WKing', 'WBishop1', 'WBishop2', 'BKing')

        Returns:
            tuple: Position as (row, col) or None if piece not set
        """
        return self.pieces.get(piece_name)
    
    
    def print_state(self):
        """
        Print the current state of the board.
        """
        board = [['.' for _ in range(8)] for _ in range(8)]
        
        for piece, position in self.pieces.items():
            if position:
                row, col = position
                if piece == 'WKing':
                    symbol = 'K'
                elif piece == 'BKing':
                    symbol = 'k'
                elif piece in ['WBishop1', 'WBishop2']:
                    symbol = 'B'
                else:
                    symbol = '?'  # Default symbol for unknown piece
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
                if piece == 'WKing':
                    symbol = 'K'
                elif piece == 'BKing':
                    symbol = 'k'
                elif piece in ['WBishop1', 'WBishop2']:
                    symbol = 'B'
                else:
                    symbol = '?'  # Default symbol for unknown piece
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
        Convert a UCI move (e.g. 'd2d4') to algebraic notation (e.g. 'Kd4', 'Bd4', etc.) using only the state.
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
                    elif name in ['WBishop1', 'WBishop2']:
                        symbol = 'B'
                    elif name == 'BKing':
                        symbol = 'k'
                    else:
                        symbol = name[0]
                    return f"{symbol}{to_square}"
        return uci_move