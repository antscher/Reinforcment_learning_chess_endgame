import chess

class State:
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

