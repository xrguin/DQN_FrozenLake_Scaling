import gymnasium as gym
import numpy as np
import pickle
import time
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics
from moviepy.editor import VideoFileClip, concatenate_videoclips
import logging

class PolicyIterationWithViz:
    def __init__(self, env_name="FrozenLake-v1", map_name="4x4", gamma=0.9):
        self.env_name = env_name
        self.map_name = map_name
        self.env = gym.make(env_name, map_name=map_name, is_slippery=True)
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.gamma = gamma
        self.policy = np.random.choice(self.n_actions, size=self.n_states)
        self.value_function = np.zeros(self.n_states)
    
    def policy_evaluation(self, theta=1e-8):
        while True:
            delta = 0
            for s in range(self.n_states):
                v = self.value_function[s]
                a = self.policy[s]
                new_value = sum(prob * (reward + self.gamma * self.value_function[next_state] * (not done))
                               for prob, next_state, reward, done in self.env.P[s][a])
                self.value_function[s] = new_value
                delta = max(delta, abs(v - self.value_function[s]))
            if delta < theta:
                break
    
    def policy_improvement(self):
        policy_stable = True
        for s in range(self.n_states):
            old_action = self.policy[s]
            action_values = [sum(prob * (reward + self.gamma * self.value_function[next_state] * (not done))
                               for prob, next_state, reward, done in self.env.P[s][a])
                           for a in range(self.n_actions)]
            self.policy[s] = np.argmax(action_values)
            if old_action != self.policy[s]:
                policy_stable = False
        return policy_stable

    def record_episode(self, episode_num, is_converged=False, video_folder="policy_frozen_lake_videos", delay=0.05):
        prefix = "converged" if is_converged else f"iteration_{episode_num}"
        env = gym.make(self.env_name, render_mode='rgb_array', is_slippery=True, map_name=self.map_name)
        env = RecordVideo(env, video_folder=video_folder, 
                         name_prefix=prefix,
                         episode_trigger=lambda x: True)
        env = RecordEpisodeStatistics(env)
        
        state, info = env.reset()
        total_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action = self.policy[state]
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            env.render()
            time.sleep(delay)
            logging.info(f"Episode {episode_num} - Step reward: {reward}")
        
        env.close()
        return total_reward, f"{video_folder}/episode_{episode_num}-episode-0.mp4"

    def train_and_record(self, max_iterations=10000, record_every=100):
        recorded_videos = []
        episode_num = 0

        for i in range(max_iterations):
            self.policy_evaluation()
            policy_stable = self.policy_improvement()
            
            if i % record_every == 0:

                reward, video_path = self.record_episode(i, is_converged=False)
                recorded_videos.append((reward, video_path))
                print(f"Iteration {i}: Recorded episode with reward {reward}")
            
            if policy_stable:
                print(f"Policy converged after {i+1} iterations")
                # Record one final episode
                reward, video_path = self.record_episode(episode_num, is_converged=True)
                recorded_videos.append((reward, video_path))
                break
        
        # Combine videos
        # clips = [VideoFileClip(video_path) for _, video_path in recorded_videos]
        # final_clip = concatenate_videoclips(clips)
        # final_clip.write_videofile("policy_frozen_lake_videos/training_progress.mp4")
        
        return recorded_videos
    
    def save_policy(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.policy, f)
    
    def load_policy(self, filename):
        with open(filename, 'rb') as f:
            self.policy = pickle.load(f)

# Example usage
if __name__ == "__main__":
    pi = PolicyIterationWithViz()
    videos = pi.train_and_record(record_every=100)
    pi.save_policy("4x4_policy.pkl")
    
    print("\nTraining completed!")
    print(f"Recorded {len(videos)} episodes")
    print("Combined video saved as 'frozen_lake_videos/training_progress.mp4'")