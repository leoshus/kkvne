import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop


class KKEvalModel(Model):
    def __init__(self, num_actions):
        super().__init__('kk_eval_model_network')
        self.layer1 = tf.keras.layers.Dense(10, activation='relu')
        self.logits = tf.keras.layers.Dense(num_actions, activation=None)
        pass

    def __call__(self, inputs):
        x = tf.convert_to_tensor(inputs)
        layer1 = self.layer1(x)
        logits = self.logits(layer1)
        return logits

    pass


class KKTargetModel(Model):
    def __init__(self, num_actions):
        super().__init__('kk_target_model_network')
        self.layer1 = tf.keras.layers.Dense(10, trainable=False, activation='relu')
        self.logits = tf.keras.layers.Dense(num_actions, trainable=False, activation=None)

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        layer1 = self.layer1(x)
        logits = self.logits(layer1)
        return logits


class DQN:
    def __init__(self, n_actions, n_features, eval_model, target_model):
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = 0.01
        self.reward_decay = 0.9
        self.e_greedy = 0.9
        self.replace_target_iter = 300
        self.memory_size = 500
        self.batch_size = 32
        self.e_greedy_increment = None

        self.learning_step_counter = 0
        self.epsilon = 0 if self.e_greedy_increment is not None else self.e_greedy
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))
        self.eval_model = eval_model
        self.target_model = target_model

        self.eval_model.compile(
            optimizer=RMSprop(lr=self.learning_rate),
            loss='mse'
        )
        self.cost_his = []

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            actions_value = self.eval_model.predict(observation)
            print(actions_value)
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = self.memory[sample_index, :]
        q_next = self.target_model.predict(batch_memory[:, -self.n_features, :])
        q_eval = self.eval_model.predict(batch_memory[:, :, self.n_features])

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.reward_decay * np.max(q_next, axis=1)

        if self.learning_step_counter % self.replace_target_iter == 0:
            for eval_layer, target_layer in zip(self.eval_model.layers, self.target_model.layers):
                target_layer.set_weights(eval_layer.get_weights())
            print('\n target_param_replaced\n')
        self.cost = self.eval_model.train_on_batch(batch_memory[:, :self.n_features], q_target)
        self.cost_his.append(self.cost)
        self.epsilon = self.epsilon + self.e_greedy_increment if self.epsilon < self.e_greedy else self.e_greedy
        self.learning_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def train(self):
        step = 0
        for episode in range(self.episodes):
            observation = env.reset()
            while True:
                env.render()
                action = RL.choose_action(observation=None)
                observation_, reward, done = env.step(action)
                RL.store_transition(observation, action, reward, observation_)
                if step > 200 and step % 5 == 0:
                    RL.learn()
                observation = observation_
                if done:
                    break
                step += 1
        env.destory()

    def _build_model(self):
        pass


if __name__ == "__main__":
    RL = DQN()

    RL.plot_cost()
