from states.state_KRvsk import State

fen = "8/8/1k6/8/8/4K3/3R4/8 w - - 0 1"
state = State()
state.create_from_fen(fen)
state.print_state()

move_stae = state.get_new_state("Rd6")
move_stae.print_state()

print(move_stae.to_fen())