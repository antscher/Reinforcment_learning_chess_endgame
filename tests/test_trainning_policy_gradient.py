import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from trainning.trainning_policy_gradient import Trainer, plot_avg_reward_and_time
from states.state_KRvsk import State
import time
import numpy as np

if __name__ == "__main__":
    trainer = Trainer()
    trainer.alpha = 0.01
    trainer.baseline = 0.0
    trainer.max_episode_length = 1000
    history_all_rewards = []

    # Parameters
    train_episodes_per_fen = 10
    num_train_fens = 100


    print(f"Training on {num_train_fens} different FENs, {train_episodes_per_fen} episodes each...")
    start = time.time()
    for fen_idx in range(num_train_fens):
        train_fen = State.random_kr_vs_k_fen()
        print(f"\nTraining on FEN {fen_idx+1}/{num_train_fens}: {train_fen}")
        history = []
        for ep in range(train_episodes_per_fen):
            root_state = State()
            root_state.create_from_fen(train_fen)
            reward, trajectory = trainer.run_episode(root_state)
            trainer.update_policy(trajectory, reward)
            history.append(reward)
        history_all_rewards.append(history)
            #print(f"  Train Episode {ep+1}/{train_episodes_per_fen} | Reward: {reward} | Length: {len(trajectory)} | Baseline: {trainer.baseline:.3f}")
    print(f"Training finished in {time.time()-start:.1f}s")

    # Save policy
    trainer.save('results/policy_gradient.pkl')
    plot_avg_reward_and_time(history_all_rewards)


