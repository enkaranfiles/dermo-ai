"""Manages the multi-turn conversation state and system prompt."""

from dataclasses import dataclass, field
from typing import Optional

from services.symptom_parser import SymptomState

DISCLAIMER = (
    "\n\n⚠️ Bu sistem yalnızca bilgilendirme amaçlıdır. "
    "Kesin teşhis için dermatoloji uzmanına başvurunuz."
)

SYSTEM_PROMPT = """Sen dermatoloji alanında bilgili, yardımsever bir sağlık bilgi asistanısın.
Görevin kullanıcının deri problemlerini anlayarak olası dermatolojik durumlar hakkında genel bilgi vermektir.

KURALLARIN:
1. Her zaman Türkçe konuş.
2. Tıbbi teşhis KOYMA — yalnızca bilgilendirme ver.
3. İlaç önerme.
4. Doktor muayenesinin yerine geçme.
5. Nazik, açık ve yönlendirici sorular sor.
6. Semptomları adım adım topla: konum, görünüm, süre, kaşıntı/ağrı/büyüme.
7. Yanıtların kısa ve anlaşılır olsun.
8. Acil durumlarda (hızla büyüyen lezyon, kanama, yüksek ateş) derhal doktora gitmelerini söyle.

YASAKLAR:
- Kesin teşhis koymak
- İlaç önermek
- Doktor görüşü gerektiren kararlar vermek"""


@dataclass
class ConversationManager:
    """Holds the full conversation history and symptom state."""

    messages: list[dict] = field(default_factory=list)
    symptom_state: SymptomState = field(default_factory=SymptomState)
    turn_count: int = 0
    diagnosis_done: bool = False
    # Stored so the diagnosis step can use the image even on later turns
    image_b64: Optional[str] = None
    image_media_type: str = "image/jpeg"
    # Stage-1 predicted group IDs (set after first group prediction)
    predicted_group_ids: list[int] = field(default_factory=list)

    def add_user_message(
        self,
        text: str,
        image_b64: Optional[str] = None,
        media_type: str = "image/jpeg",
    ) -> None:
        if image_b64:
            # Build multimodal content so Claude sees the image in every turn
            content: list[dict] = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": text},
            ]
            self.messages.append({"role": "user", "content": content})
            # Persist for later diagnosis call
            self.image_b64 = image_b64
            self.image_media_type = media_type
        else:
            self.messages.append({"role": "user", "content": text})
        self.turn_count += 1

    def add_assistant_message(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

    def get_messages(self) -> list[dict]:
        return self.messages

    def get_system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def build_greeting(self) -> str:
        return (
            "Merhaba! Ben dermatoloji bilgi asistanınızım. "
            "Deri probleminiz hakkında size genel bilgi sunmaya çalışacağım.\n\n"
            "Lütfen şikayetinizi anlatın: Nerede bir sorun var ve nasıl görünüyor?"
            + DISCLAIMER
        )

    # Group-specific targeted questions (group_id → question list)
    # Each entry targets intra-group differential: most discriminating questions first.
    _GROUP_QUESTIONS: dict[int, list[str]] = {

        # ── 1. Akne ve Rozasea ──────────────────────────────────────────
        # Akne vulgaris vs Rozasea
        1: [
            "• Sivilceler iltihaplı mı (sarı/beyaz başlı)?",           # akne → püstül/papül
            "• Yüzde sürekli kızarıklık veya yanma hissi var mı?",     # rozasea → persistan eritem
            "• Yüzde ince kırmızı damarlar belirgin mi?",              # rozasea → telenjiektazi
        ],

        # ── 2. Aktinik Keratoz / BHK / SHK ─────────────────────────────
        2: [
            "• Lezyon güneşe maruz kalan bir bölgede mi?",             # hepsi → UV ilişkisi
            "• Yara uzun süredir iyileşmiyor mu?",                     # BHK/SHK → kronik ülser
            "• Lezyon hızlı büyüyor mu?",                              # SHK → agresif seyir
            "• Lezyon yüzeyi pürüzlü veya zımpara gibi mi?",          # aktinik keratoz → keratotik yüzey
        ],

        # ── 3. Atopik Dermatit ──────────────────────────────────────────
        3: [
            "• Kaşıntı geceleri daha mı kötüleşiyor?",                # nokturnal pruritus
            "• Ailede egzama, astım veya alerjik rinit var mı?",       # atopik triad
            "• Döküntü dirsek iç kıvrımı veya diz arkasında mı?",     # fleksural dağılım
            "• Şikayetler tekrarlayıcı mı (alevlenip söner mi)?",     # kronik-relaps seyir
        ],

        # ── 4. Büllöz Hastalıklar ──────────────────────────────────────
        # Pemfigus vulgaris vs Büllöz pemfigoid
        4: [
            "• Deri üzerinde su dolu bül/kabarcık var mı?",            # her ikisi → bül
            "• Kabarcıklar kolayca patlıyor mu, gevşek mi?",           # pemfigus → gevşek bül
            "• Ağız içinde yara veya ülser var mı?",                   # pemfigus → mukozal tutulum
            "• Kabarcıklar gergin ve zor patlıyor mu?",                # büllöz pemfigoid → gergin bül
        ],

        # ── 5. Selülit / İmpetigo / Follikülit ─────────────────────────
        5: [
            "• Etkilenen bölge kızarık, şişmiş ve sıcak mı?",         # selülit → derin enfeksiyon
            "• Ateş veya titreme var mı?",                             # selülit → sistemik belirtiler
            "• Bal rengi kabuklanma var mı?",                          # impetigo → tipik kabuk
            "• Döküntü kıl diplerinde mi yerleşmiş?",                  # follikülit → foliküler dağılım
        ],

        # ── 6. Egzama ──────────────────────────────────────────────────
        # Nümmüler vs Dishidrotik
        6: [
            "• Kaşıntı yoğun mu ve hangi bölgede?",                   # genel egzama
            "• Döküntü madeni para şeklinde yuvarlak mı?",             # nümmüler → disk şekli
            "• El ayası veya parmak kenarlarında küçük kabarcıklar var mı?",  # dishidrotik → vezikül
        ],

        # ── 7. Egzantemler ve İlaç Erüpsiyonları ────────────────────────
        7: [
            "• Yakın zamanda yeni ilaç kullanmaya başladınız mı?",     # ilaç erüpsiyonu → temporal ilişki
            "• Döküntü tüm vücuda yayıldı mı?",                       # her ikisi → yaygın dağılım
            "• Döküntüden önce ateş, boğaz ağrısı veya halsizlik oldu mu?",  # viral → prodrom
        ],

        # ── 8. Saç Dökülmesi / Alopesi ─────────────────────────────────
        # Alopesi areata vs Androgenetik alopesi
        8: [
            "• Saç dökülmesi yuvarlak bir alanda mı, yoksa yaygın mı?",  # areata → lokalize; androgenetik → diffüz
            "• Saç dökülmesi ani mi başladı?",                           # areata → akut başlangıç
            "• Saç çizgisi geriliyor veya tepe bölgesi seyreliyor mu?",  # androgenetik → pattern
        ],

        # ── 9. Herpes / HPV / STD ──────────────────────────────────────
        9: [
            "• Lezyon ağrılı, küme halinde küçük kabarcıklar mı?",    # herpes → gruplu vezikül
            "• Benzer şikayetler daha önce aynı bölgede tekrarladı mı?",  # herpes → nüks
            "• Lezyon karnabahar görünümünde ağrısız bir çıkıntı mı?", # kondilom → verrüköz papül
            "• Lezyon genital veya dudak/ağız çevresinde mi?",         # lokalizasyon → alt tip ayrımı
        ],

        # ── 10. Pigmentasyon Bozuklukları ──────────────────────────────
        # Vitiligo vs Melazma
        10: [
            "• Leke beyaz (renksiz) mi, yoksa koyu kahverengi mi?",   # vitiligo → depigmentasyon; melazma → hiperpigmentasyon
            "• Leke güneşte daha belirgin hale geliyor mu?",           # melazma → UV ile koyulaşma
            "• Beyaz lekeler simetrik mi dağılmış?",                   # vitiligo → simetrik dağılım
        ],

        # ── 11. Lupus ve Bağ Dokusu Hastalıkları ───────────────────────
        11: [
            "• Güneşe çıkınca döküntü şiddetleniyor mu?",             # her ikisi → fotosensitivite
            "• Yanaklarda kelebek şeklinde kızarıklık var mı?",        # lupus → malar rash
            "• Eklem ağrısı veya kas güçsüzlüğü var mı?",             # lupus → artralji; DM → miyopati
            "• Göz kapağında mor renk değişikliği var mı?",            # dermatomiyozit → heliotrope
        ],

        # ── 12. Melanom / Nevüs — ABCDE kriterleri ────────────────────
        12: [
            "• Lezyon asimetrik mi ve sınırları düzensiz mi?",         # melanom → A + B
            "• Renk içinde farklı tonlar var mı (kahverengi, siyah, kırmızı)?",  # melanom → C
            "• Lezyon çapı 6 mm'den büyük mü?",                       # melanom → D
            "• Lezyon son zamanlarda büyüdü, renk veya şekil değiştirdi mi?",  # melanom → E (evolving)
        ],

        # ── 13. Tırnak Hastalıkları ────────────────────────────────────
        13: [
            "• Tırnak kalınlaşması veya renk değişikliği var mı?",    # onikomikoz → distrofik tırnak
            "• Tırnak çevresi şişmiş ve ağrılı mı?",                  # paronişi → periungual iltihap
            "• Tırnak altında kırıntı birikimi var mı?",              # onikomikoz → subungual debris
        ],

        # ── 14. Kontakt Dermatit ───────────────────────────────────────
        # Alerjik vs İrritan
        14: [
            "• Döküntü belirli bir maddeyle temas ettikten sonra mı çıktı?",  # her ikisi → temas öyküsü
            "• Döküntü bölgesinde küçük kabarcıklar veya sızıntı var mı?",    # alerjik → vezikül
            "• Temas bölgesinde kuruluk, çatlak ve yanma mı var?",            # irritan → kuru dermatit
            "• Eldiven, takı, kozmetik veya temizlik maddesi kullanıyor musunuz?",  # tetikleyici ajanlar
        ],

        # ── 15. Psoriazis / Liken Planus ──────────────────────────────
        15: [
            "• Lezyonlar üzerinde gümüşi beyaz pullanma var mı?",     # psoriazis → skuam
            "• Lezyonlar mor renkli ve parlak yüzeyli mi?",            # liken planus → poligonal papül
            "• Eklem ağrısı veya sabah tutukluğu var mı?",             # psöriatik artrit
            "• Ağız içinde beyaz ağsı çizgiler var mı?",               # liken planus → oral Wickham
        ],

        # ── 16. Skabiyez / Böcek Isırığı ──────────────────────────────
        16: [
            "• Kaşıntı geceleri çok artıyor mu?",                     # skabiyez → nokturnal pruritus
            "• Aile bireylerinde veya yakın temaslıda benzer kaşıntı var mı?",  # skabiyez → bulaş
            "• Parmak araları, bilek veya kasıkta mı kaşıntı?",       # skabiyez → predileksiyon bölgeleri
            "• Ciltte böcek ısırığına benzer tek tek şişlikler mi var?",  # böcek ısırığı → izole papüller
        ],

        # ── 17. Seboreik Keratoz / Dermatofibrom / Lipom ──────────────
        17: [
            "• Lezyon cildin üstüne yapışık gibi mi duruyor?",        # seb. keratoz → 'stuck-on'
            "• Lezyon sert mi ve sıkıştırınca ortası çukurlaşıyor mu?",  # dermatofibrom → dimple sign
            "• Lezyon cilt altında, yumuşak ve hareketli mi?",         # lipom → subkutan, mobil
            "• Lezyon yavaş yavaş mı büyüdü?",                        # hepsi → yavaş seyir
        ],

        # ── 18. Sistemik Hastalık ──────────────────────────────────────
        18: [
            "• Diyabet veya başka kronik hastalığınız var mı?",        # diyabetik dermopati → DM öyküsü
            "• Bacağın ön yüzünde kahverengi yuvarlak lekeler var mı?",  # diyabetik dermopati → pretibial
        ],

        # ── 19. Tinea / Kandidiyazis / Fungal Enfeksiyonlar ───────────
        19: [
            "• Lezyon halka şeklinde mi?",                             # tinea korporis → anüler plak
            "• Parmak araları, kasık veya kıvrım bölgelerinde mi?",   # kandidiyazis / tinea pedis → intertriginöz
            "• Nemli ortamlarda (havuz, spor salonu) zaman geçiriyor musunuz?",  # risk faktörü
            "• Lezyonun kenarında küçük uydu püstüller var mı?",       # kandidiyazis → satellit lezyon
        ],

        # ── 20. Ürtiker ────────────────────────────────────────────────
        # Akut vs Kronik
        20: [
            "• Kabarıklıklar yer değiştiriyor mu?",                   # ürtiker → gezici plak
            "• Her bir kabarıklık 24 saat içinde kayboluyor mu?",     # ürtiker → geçici lezyon
            "• Şikayetler 6 haftadan uzun süredir devam ediyor mu?",  # kronik ürtiker eşiği
            "• Dudak veya göz kapağında şişme oluyor mu?",             # anjioödem riski
        ],

        # ── 21. Vasküler Tümörler ──────────────────────────────────────
        # Hemanjiom vs Piyojenik granülom
        21: [
            "• Lezyon kırmızı veya mor renkli mi?",                   # her ikisi → vasküler görünüm
            "• Basınçla rengi soluyor mu?",                            # hemanjiom → blanching
            "• Lezyon çok kolay kanıyor mu?",                          # piyojenik granülom → frajil
            "• Lezyon bebeklik döneminde mi, yoksa travma sonrası mı çıktı?",  # hemanjiom → infantil; PG → travma
        ],

        # ── 22. Vaskülit ──────────────────────────────────────────────
        22: [
            "• Bacak veya ayakta mor-kırmızı noktalar var mı?",       # palpabl purpura
            "• Bu noktalar basınçla soluyor mu?",                      # vaskülit purpurası → solmaz
            "• Eklem ağrısı, karın ağrısı veya halsizlik var mı?",    # sistemik tutulum taraması
        ],

        # ── 23. Siğil / Molluskum / Viral Enfeksiyonlar ───────────────
        23: [
            "• Lezyonun ortası çukur (göbekli) mu?",                  # molluskum → umblike papül
            "• Yüzey pürüzlü ve sert mi?",                            # siğil → verrüköz yapı
            "• Sıkıştırınca beyaz, peynir gibi madde çıkıyor mu?",   # molluskum → kazein benzeri core
        ],
    }

    def build_follow_up_questions(self) -> str:
        """
        Generate targeted follow-up questions.
        If group prediction has run, ask group-specific clinical questions.
        Otherwise fall back to generic slot-filling questions.
        """
        missing = self.symptom_state.missing_slots()
        questions: list[str] = []

        # Always ask for location first if missing
        if "location" in missing:
            questions.append("• Şikayetiniz vücudunuzun hangi bölgesinde?")

        if self.predicted_group_ids:
            # Collect group-specific questions (deduplicated, max 3 total)
            seen: set[str] = set()
            for gid in self.predicted_group_ids:
                for q in self._GROUP_QUESTIONS.get(gid, []):
                    if q not in seen:
                        seen.add(q)
                        questions.append(q)
                        if len(questions) >= 3:
                            break
                if len(questions) >= 3:
                    break
        else:
            # Generic fallback
            if "symptoms" in missing:
                questions.append("• Kaşıntı, ağrı veya yanma var mı?")
            if "duration" in missing:
                questions.append("• Bu sorun ne zamandır var?")
            if "appearance" in missing:
                questions.append("• Lezyonun rengi ve şekli nasıl? (Kızarık, koyu, kabarcıklı, vb.)")

        if not questions:
            questions.append("• Başka eklemek istediğiniz bir şey var mı?")

        return "Birkaç soru sormam gerekiyor:\n" + "\n".join(questions)

    def should_proceed_to_diagnosis(self) -> bool:
        """Decide whether we have enough info to run the diagnosis pipeline."""
        return self.symptom_state.is_sufficient() and self.turn_count >= 2

    def format_disclaimer(self) -> str:
        return DISCLAIMER
