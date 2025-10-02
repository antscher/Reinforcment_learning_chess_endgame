# Reinforcment_learning_chess_endgame
In this repository, we want to devellop an IA bot for endgame problems such as KRvsk or KBBvs k. In this approach, we will use Qlearning to train againt a real bot


# Organisation of the repo:
- `states/` contains the build environment (state logic, QTable, etc.)
- `stockfish/` contains the Stockfish chess engine binary (bot not from us)
- `tests/` contains all test scripts
- `results/` contains our save results from test founctions
- training : our actual algorithm

  
# End of state : 
- Rule of repetition + 50 turn 
- Mate or draw

## Actual rate  trainning
for a mate in 1 train in approwimatelly 40 episodes
for a mate in 2 train in approwimatelly 400 episodes (50 with previous trainning on the 1 mate intermediate (equal to mate in 1))
Idea : decompose in rush of 50 episode on 1 fen, also do double to tenth Qlearnig with multithread to make the learning faster

# For testing
run file in the test folder
you need first to intall : "pip install chess"
try just to run "test_training.py" for running the Q_table

# Idea for making it faster : 
- start with mate in 1
- Use symetry (mirror, exactly the same state) (added)
- Multiple Qlearning

