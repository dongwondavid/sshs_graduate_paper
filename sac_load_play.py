# SAC load and play (tf2 subclassing API version)
# coded by St.Watermelon

import gym
from sac_learn2 import SACagent
import tensorflow as tf

def main():

    env = gym.make("Pendulum-v1")
    agent = SACagent(env)

    agent.load_weights('./save_weights/')

    time = 0
    state,_ = env.reset()

    while True:
        env.render()
        action = agent.actor(tf.convert_to_tensor([state], dtype=tf.float32))[0][0]
        state, reward, info, done, _ = env.step(action)
        time += 1

        print('Time: ', time, 'Reward: ', reward)

        if done:
            break

    env.close()

if __name__=="__main__":
    main()