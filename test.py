import chess
from states import State


"""
import chess.engine

engine = chess.engine.SimpleEngine.popen_uci(r"stockfish\stockfish-windows-x86-64-avx2.exe")

board = chess.Board()
while not board.is_game_over():
    result = engine.play(board, chess.engine.Limit(time=0.1))
    board.push(result.move)

engine.quit()
"""

#Create a chess board from a FEN string
board = chess.Board("8/8/1k6/8/8/4K3/3R4/8 w - - 0 1")
#Print the board
print(board)

Nf3 = chess.Move.from_uci("g1f3")
board.push(Nf3)  # Make the move

board.pop()  # Unmake the last move

# Check if the game is over
board.is_stalemate()
board.is_insufficient_material()
board.outcome()

#detection of end due to other rules
board.can_claim_fifty_moves() #boolean
board.halfmove_clock #number of halfmoves since last capture or pawn move
board.can_claim_threefold_repetition() #boolean

#Get legal move in UCI
board.legal_moves


