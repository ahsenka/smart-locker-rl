import random

import numpy as np


class QLearningAgent:
    def __init__(
        self,
        action_count=5,
        alpha=0.1,
        gamma=0.92,
        epsilon=0.35,
        min_epsilon=0.02,
        epsilon_decay=0.996,
        seed=None,
    ):
        self.action_count = action_count
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.random = random.Random(seed)
        self.q_table = {}

    def create_q_values(self, state):
        # Daha once gorulmeyen state icin q degerleri sifirdan basliyor.
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_count)

    def select_action(self, state, training=True, valid_actions=None):
        self.create_q_values(state)
        action_pool = valid_actions or list(range(self.action_count))

        # Egitimde bazen kesif yapmasi icin rastgele aksiyon sectiriyorum.
        if training and self.random.random() < self.epsilon:
            return self.random.choice(action_pool)

        q_values = self.q_table[state]
        return max(action_pool, key=lambda action: q_values[action])

    def learn(self, state, action, reward, next_state, done=False):
        self.create_q_values(state)
        self.create_q_values(next_state)

        old_value = self.q_table[state][action]
        future_value = 0 if done else np.max(self.q_table[next_state])

        # Klasik q-learning guncellemesi.
        new_value = old_value + self.alpha * (
            reward + self.gamma * future_value - old_value
        )

        self.q_table[state][action] = new_value

    def update_epsilon(self):
        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.epsilon_decay,
        )

    # Eski fonksiyon isimleri bozulmasin diye biraktim.
    def aksiyon_sec(self, state):
        return self.select_action(state)

    def ogren(self, state, aksiyon, odul, sonraki_state):
        self.learn(state, aksiyon, odul, sonraki_state)

    def epsilon_guncelle(self):
        self.update_epsilon()


class RandomAgent:
    def __init__(self, action_count=5, seed=None):
        self.action_count = action_count
        self.random = random.Random(seed)

    def select_action(self, state, training=False, valid_actions=None):
        if valid_actions:
            return self.random.choice(valid_actions)
        return self.random.randint(0, self.action_count - 1)


class RuleBasedAgent:
    def select_action(self, state, training=False, valid_actions=None):
        (
            empty_small,
            empty_medium,
            empty_large,
            empty_xlarge,
            empty_cold,
            package_size,
            priority,
            cold_required,
            fragile,
        ) = state

        empty = {
            0: empty_small,
            1: empty_medium,
            2: empty_large,
            3: empty_xlarge,
            4: empty_cold,
        }

        if cold_required and empty[4] > 0 and package_size <= 2:
            return 4

        for action in range(package_size, 4):
            if empty[action] > 0:
                return action

        if cold_required and fragile and empty[4] > 0 and package_size <= 2:
            return 4

        for action in range(4, -1, -1):
            if empty[action] > 0:
                return action

        return 0
