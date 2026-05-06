import random


class LockerEnvironment:

    def __init__(self):

        # toplam locker sayıları
        self.max_kucuk = 5
        self.max_orta = 5
        self.max_buyuk = 5

        self.reset()

    def reset(self):

        # boş locker sayıları başlangıçta full oluyor
        self.bos_kucuk = self.max_kucuk
        self.bos_orta = self.max_orta
        self.bos_buyuk = self.max_buyuk

        # ilk paket oluşturuluyor
        self.gelen_paket = self.rastgele_paket_uret()

        return self.state_getir()

    def rastgele_paket_uret(self):

        # küçük paketler daha sık geliyor
        return random.choices(
            population=[0, 1, 2],
            weights=[0.5, 0.3, 0.2],
            k=1
        )[0]

    def state_getir(self):

        return (
            self.bos_kucuk,
            self.bos_orta,
            self.bos_buyuk,
            self.gelen_paket
        )

    def uygun_mu(self, aksiyon):

        # 0 = küçük
        # 1 = orta
        # 2 = büyük

        # paket sadece kendinden büyük veya eşit lockera girebilir
        return aksiyon >= self.gelen_paket

    def step(self, aksiyon):

        odul = 0
        done = False

        # geçersiz aksiyon kontrolü
        if not self.uygun_mu(aksiyon):

            odul = -20

            return self.state_getir(), odul, done

        # küçük lockera koy
        if aksiyon == 0:

            if self.bos_kucuk <= 0:

                odul = -20

                return self.state_getir(), odul, done

            self.bos_kucuk -= 1

        # orta lockera koy
        elif aksiyon == 1:

            if self.bos_orta <= 0:

                odul = -20

                return self.state_getir(), odul, done

            self.bos_orta -= 1

        # büyük lockera koy
        elif aksiyon == 2:

            if self.bos_buyuk <= 0:

                odul = -20

                return self.state_getir(), odul, done

            self.bos_buyuk -= 1

        # tam uygun yerleşim
        if aksiyon == self.gelen_paket:

            odul = 10

        # bir üst boy locker kullanıldı
        elif aksiyon == self.gelen_paket + 1:

            odul = 3

        # gereksiz büyük kullanım
        else:

            odul = 1

        # büyük lockeri küçük paketle işgal etme cezası
        if aksiyon == 2 and self.gelen_paket != 2:

            odul -= 5

        # tüm lockerlar dolduysa episode bitiyor
        if (
            self.bos_kucuk == 0 and
            self.bos_orta == 0 and
            self.bos_buyuk == 0
        ):

            done = True

        # yeni paket geliyor
        self.gelen_paket = self.rastgele_paket_uret()

        return self.state_getir(), odul, done