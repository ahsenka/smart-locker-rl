import random
import numpy as np


class QLearningAgent:

    def __init__(self):

        # öğrenme oranı
        self.alpha = 0.1

        # gelecekteki ödül etkisi
        self.gamma = 0.9

        # keşif oranı
        self.epsilon = 0.2

        # q tablosu
        self.q_tablosu = {}
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.995

    def q_degerlerini_olustur(self, state):

        # state daha önce yoksa q değerleri oluşturuluyor
        if state not in self.q_tablosu:

            self.q_tablosu[state] = np.zeros(3)

    def aksiyon_sec(self, state):

        self.q_degerlerini_olustur(state)

        # exploration
        if random.uniform(0, 1) < self.epsilon:

            return random.randint(0, 2)

        # exploitation
        return np.argmax(self.q_tablosu[state])

    def ogren(self, state, aksiyon, odul, sonraki_state):

        self.q_degerlerini_olustur(state)
        self.q_degerlerini_olustur(sonraki_state)

        eski_deger = self.q_tablosu[state][aksiyon]

        en_iyi_sonraki = np.max(
            self.q_tablosu[sonraki_state]
        )

        # klasik q-learning formülü
        yeni_deger = eski_deger + self.alpha * (
            odul + self.gamma * en_iyi_sonraki - eski_deger
        )

        self.q_tablosu[state][aksiyon] = yeni_deger
    def epsilon_guncelle(self):

        self.epsilon = max(
        self.min_epsilon,
        self.epsilon * self.epsilon_decay
    )