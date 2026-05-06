from src.environment import LockerEnvironment
from src.agent import QLearningAgent
import random


class Trainer:

    def __init__(self):

        self.env = LockerEnvironment()
        self.agent = QLearningAgent()

        # 10000 yerine 5000 seçildi
        # bu problemde state sayısı düşük olduğu için yeterli oluyor
        self.episode_sayisi = 5000
        self.max_step = 30

        # grafikler için tutuluyor
        self.toplam_oduller = []
        self.basarili_episode = []
        self.random_oduller = []

    def egit(self):

        for episode in range(self.episode_sayisi):

            state = self.env.reset()
            toplam_odul = 0

            for step in range(self.max_step):

                # ajan epsilon-greedy ile aksiyon seçiyor
                aksiyon = self.agent.aksiyon_sec(state)

                # ortam aksiyonu uyguluyor
                sonraki_state, odul, done = self.env.step(aksiyon)

                # q tablosu güncelleniyor
                self.agent.ogren(
                    state,
                    aksiyon,
                    odul,
                    sonraki_state
                )

                state = sonraki_state
                toplam_odul += odul

                if done:
                    break

            self.toplam_oduller.append(toplam_odul)

            if toplam_odul > 0:
                self.basarili_episode.append(1)
            else:
                self.basarili_episode.append(0)

            # episode sonunda keşif oranı azaltılıyor
            self.agent.epsilon_guncelle()

            if (episode + 1) % 100 == 0:
                print(
                    f"episode: {episode + 1} | "
                    f"toplam ödül: {toplam_odul} | "
                    f"epsilon: {self.agent.epsilon:.3f}"
                )

        print("\n eğitim tamamlandı \n")

        return self.agent, self.toplam_oduller

    def random_agent_test_et(self):

        for episode in range(self.episode_sayisi):

            state = self.env.reset()
            toplam_odul = 0

            for step in range(self.max_step):

                # random agent tamamen rastgele karar veriyor
                aksiyon = random.randint(0, 2)

                sonraki_state, odul, done = self.env.step(aksiyon)

                state = sonraki_state
                toplam_odul += odul

                if done:
                    break

            self.random_oduller.append(toplam_odul)

        print("\n random agent testi tamamlandı \n")

        return self.random_oduller