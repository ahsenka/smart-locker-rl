from src.trainer import Trainer
from src.visualizer import Visualizer


def test_simulasyonu(env, agent, visualizer):

    state = env.reset()

    frame_listesi = []
    toplam_test_odulu = 0
    print("\n test simulasyonu başladı \n")

    for step in range(15):

        # en iyi aksiyon seçiliyor
        aksiyon = agent.aksiyon_sec(state)

        sonraki_state, odul, done = env.step(aksiyon)
        toplam_test_odulu += odul #sonda toplam odulu hesaplams olmasi icin
        bos_kucuk = env.bos_kucuk
        bos_orta = env.bos_orta
        bos_buyuk = env.bos_buyuk

        # frame oluşturuluyor
        frame = visualizer.locker_gorseli_olustur(
            bos_kucuk,
            bos_orta,
            bos_buyuk,
            step,
            state[3],
            aksiyon,
            odul,
            toplam_test_odulu if step == 14 or done else None
        )

        frame_listesi.append(frame)

        print(f"step: {step + 1}")
        print(f"state: {state}")
        aksiyon_isimleri = {
            0: "küçük locker",
            1: "orta locker",
            2: "büyük locker"
        }
        paket_isimleri = {
            0: "küçük paket",
            1: "orta paket",
            2: "büyük paket"
        }

        print(f"gelen paket: {paket_isimleri[state[3]]}")
        print(f"seçilen aksiyon: {aksiyon_isimleri[aksiyon]}")
        print(f"ödül: {odul}")
        print("-----------------------")

        state = sonraki_state

        if done:
            print("tüm lockerlar doldu")
            break

    # gif oluşturuluyor
    visualizer.gif_olustur(frame_listesi)

    print("\n gif oluşturuldu \n")


def main():

    trainer = Trainer()

    # eğitim başlıyor
    agent, toplam_oduller = trainer.egit()

    random_oduller = trainer.random_agent_test_et()

    basari_listesi = trainer.basarili_episode
    visualizer = Visualizer()

    # grafikler oluşturuluyor
    visualizer.odul_grafigi_ciz(toplam_oduller)
    visualizer.basari_grafigi(basari_listesi)
    visualizer.moving_average_grafigi(toplam_oduller)
    visualizer.karsilastirma_grafigi(toplam_oduller,random_oduller)
    # test simulasyonu
    test_simulasyonu(
        trainer.env,
        agent,
        visualizer
    )

    print("\n tüm işlemler tamamlandı \n")


if __name__ == "__main__":
    main()