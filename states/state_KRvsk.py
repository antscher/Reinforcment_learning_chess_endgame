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
    
    def position_to_string(self, position):
        """
        Convert a position tuple to chess notation string.
        
        Args:
            position (tuple): Position as (row, col)
            
        Returns:
            str: Position in chess notation (e.g., 'e4')
        """
        if position is None:
            return None
        
        row, col = position
        if not (0 <= row <= 7 and 0 <= col <= 7):
            raise ValueError(f"Invalid position: {position}")
        
        # Convert to chess notation: columns a-h, rows 1-8
        col_letter = chr(ord('a') + col)
        row_number = str(row + 1)
        return col_letter + row_number
    
    def string_to_position(self, position_string):
        """
        Convert a chess notation string to position tuple.
        
        Args:
            position_string (str): Position in chess notation (e.g., 'e4')
            
        Returns:
            tuple: Position as (row, col)
        """
        if position_string is None or len(position_string) != 2:
            raise ValueError(f"Invalid position string: {position_string}")
        
        col_letter = position_string[0].lower()
        row_number = position_string[1]
        
        if not ('a' <= col_letter <= 'h' and '1' <= row_number <= '8'):
            raise ValueError(f"Invalid position string: {position_string}")
        
        col = ord(col_letter) - ord('a')
        row = int(row_number) - 1
        
        return (row, col)
    
    def get_positions_as_strings(self):
        """
        Get all piece positions as chess notation strings.
        
        Returns:
            dict: Dictionary with piece names and their positions as strings
        """
        string_positions = {}
        for piece, position in self.pieces.items():
            string_positions[piece] = self.position_to_string(position)
        return string_positions
    
    def set_position_from_string(self, piece_name, position_string):
        """
        Set a piece position using chess notation string.
        
        Args:
            piece_name (str): Name of the piece ('WKing', 'WRook', 'BKing')
            position_string (str): Position in chess notation (e.g., 'e4')
        """
        position = self.string_to_position(position_string)
        self.update(piece_name, position)
    
    def __str__(self):
        """String representation of the state."""
        string_positions = self.get_positions_as_strings()
        return f"State(WKing: {string_positions['WKing']}, WRook: {string_positions['WRook']}, BKing: {string_positions['BKing']})"
    
    def __repr__(self):
        """String representation for debugging."""
        return f"State({self.pieces})"
