import os
import imageio
import matplotlib.pyplot as plt


class Visualizer:

    def __init__(self):

        os.makedirs("outputs/plots", exist_ok=True)
        os.makedirs("outputs/gifs", exist_ok=True)

    def odul_grafigi_ciz(self, toplam_oduller):

        plt.figure(figsize=(12, 6))

        plt.plot(toplam_oduller)

        plt.title("episode bazlı toplam ödül")
        plt.xlabel("episode")
        plt.ylabel("toplam ödül")

        plt.grid(True)

        plt.savefig(
            "outputs/plots/odul_grafigi.png"
        )

        plt.close()

    def moving_average_grafigi(self, toplam_oduller):

        pencere = 100

        ortalamalar = []

        for i in range(len(toplam_oduller)):

            baslangic = max(0, i - pencere)

            pencere_verisi = toplam_oduller[
                baslangic:i + 1
            ]

            ortalamalar.append(
                sum(pencere_verisi) / len(pencere_verisi)
            )

        plt.figure(figsize=(12, 6))

        plt.plot(ortalamalar)

        plt.title("hareketli ortalama ödül grafiği")
        plt.xlabel("episode")
        plt.ylabel("ortalama ödül")

        plt.grid(True)

        plt.savefig(
            "outputs/plots/moving_average.png"
        )

        plt.close()

    def locker_gorseli_olustur(self,bos_kucuk,bos_orta,bos_buyuk,step,gelen_paket,aksiyon,odul):

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.set_xlim(0, 6)
        ax.set_ylim(0, 4)

        ax.set_title(
            f"locker durumu - step {step}"
        )

        ax.axis("off")
        paket_isimleri = {
            0: "küçük paket",
            1: "orta paket",
            2: "büyük paket"
        }

        aksiyon_isimleri = {
            0: "küçük locker",
            1: "orta locker",
            2: "büyük locker"
        }
            # locker çizim helperı
        def locker_satiri_ciz(y,bos_sayi,toplam, label):

            for i in range(toplam):

                dolu_mu = i < (toplam - bos_sayi)

                renk = (
                    "red"
                    if dolu_mu
                    else "lightgray"
                )

                kare = plt.Rectangle(
                    (i + 0.5, y),
                    0.8,
                    0.8,
                    color=renk,
                    ec="black"
                )

                ax.add_patch(kare)

            ax.text(
                0,
                y + 0.3,
                label,
                fontsize=12,
                fontweight="bold"
            )

        locker_satiri_ciz(
            3,
            bos_kucuk,
            5,
            "küçük"
        )

        locker_satiri_ciz(
            2,
            bos_orta,
            5,
            "orta"
        )

        locker_satiri_ciz(
            1,
            bos_buyuk,
            5,
            "büyük"
        )

        dosya_adi = (
            f"outputs/gifs/frame_{step}.png"
        )
        ax.text(
            0.5,
            0.3,
            f"gelen paket: {paket_isimleri[gelen_paket]}",
            fontsize=11
        )

        ax.text(
            0.5,
            0.0,
            f"ajan kararı: {aksiyon_isimleri[aksiyon]}",
            fontsize=11
        )

        ax.text(
            3.5,
            0.0,
            f"ödül: {odul}",
            fontsize=11,
            fontweight="bold"
        )
        plt.savefig(dosya_adi)

        plt.close()

        return dosya_adi
       
       

    def gif_olustur(self, frame_listesi):

        kareler = []

        for frame in frame_listesi:

            kareler.append(
                imageio.imread(frame)
            )

        imageio.mimsave(
            "outputs/gifs/locker_simulasyonu.gif",
            kareler,
            duration=0.6
        )
    def basari_grafigi(self, basari_listesi):

        oranlar = []

        pencere = 100

        for i in range(len(basari_listesi)):

            baslangic = max(0, i - pencere)

            pencere_verisi = basari_listesi[
                baslangic:i + 1
            ]

            oran = (
                sum(pencere_verisi)
                / len(pencere_verisi)
            )

            oranlar.append(oran)

        plt.figure(figsize=(12, 6))

        plt.plot(oranlar)

        plt.title("başarı oranı")
        plt.xlabel("episode")
        plt.ylabel("başarı")

        plt.grid(True)

        plt.savefig(
            "outputs/plots/basari_grafigi.png"
        )

        plt.close()
    
    def karsilastirma_grafigi(self, q_oduller, random_oduller):

        pencere = 100

        q_ortalamalar = []
        random_ortalamalar = []

        for i in range(len(q_oduller)):

            baslangic = max(0, i - pencere)

            q_pencere = q_oduller[baslangic:i + 1]
            random_pencere = random_oduller[baslangic:i + 1]

            q_ortalamalar.append(
                sum(q_pencere) / len(q_pencere)
            )

            random_ortalamalar.append(
                sum(random_pencere) / len(random_pencere)
            )

        plt.figure(figsize=(12, 6))

        plt.plot(q_ortalamalar, label="q-learning agent")
        plt.plot(random_ortalamalar, label="random agent")

        plt.title("q-learning agent vs random agent")
        plt.xlabel("episode")
        plt.ylabel("ortalama ödül")

        plt.legend()
        plt.grid(True)

        plt.savefig(
            "outputs/plots/q_vs_random.png"
        )

        plt.close()