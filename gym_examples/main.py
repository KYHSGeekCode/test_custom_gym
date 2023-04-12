from collections import deque

import numpy as np

from dqn import DQNAgent
from envs import GridWorldEnv
import pdb


def main():
    # pdb.set_trace()
    env = GridWorldEnv("human")

    env.reset()
    state_dim = env.observation_space["agent"].shape[0] + env.observation_space["target"].shape[0]
    action_dim = env.action_space.n

    print(f"{state_dim=} {action_dim=}")

    hidden_dim = 32
    buffer_size = 30
    batch_size = 20
    gamma = 0.95
    learning_rate = 0.001

    agent = DQNAgent(
        state_dim, action_dim, hidden_dim, buffer_size, batch_size, gamma, learning_rate
    )

    num_episodes = 1000
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    max_steps_per_episode = 1000
    update_frequency = 50  # update target network every 100 steps

    scores = deque(maxlen=200)

    for episode in range(num_episodes):
        state, info = env.reset()
        state_agent = state["agent"]
        state_target = state["target"]
        state_vec = np.concatenate((state_agent, state_target)).astype('float32')
        print(state)
        episode_reward = 0

        for step in range(max_steps_per_episode):
            action = agent.select_action(state_vec, epsilon)
            next_state, reward, done, truncated, info = env.step(action)
            reward = reward - 0.1
            episode_reward += reward

            next_state_vec = np.concatenate((next_state["agent"], next_state["target"])).astype('float32')

            agent.add_experience(state_vec, action, next_state_vec, reward, done)
            agent.update_model()

            if step % update_frequency == 0:
                agent.update_target_model()  # important!

            if done:
                break

            state = next_state
            if done or truncated:
                state, info = env.reset()
                state_agent = state["agent"]
                state_target = state["target"]
                state_vec = np.concatenate((state_agent, state_target))

        scores.append(episode_reward)
        print(
            f"Episode {episode}: score={episode_reward:.2f}, avg_score={np.mean(scores):.2f}, eps={epsilon:.2f}"
        )
        epsilon = max(epsilon_min, epsilon_decay * epsilon)

        if np.mean(scores) >= 195.0 and episode >= 100:
            print(f"Solved in {episode} episodes!")
            break
    env.close()


if __name__ == '__main__':
    main()
