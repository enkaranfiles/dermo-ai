"""Claude API client wrapper for the dermatology chatbot."""

import base64
import json
from pathlib import Path
from typing import Optional

import anthropic

MODEL = "claude-haiku-4-5"

client = anthropic.Anthropic()


def chat_with_claude(
    messages: list[dict],
    system_prompt: str,
    max_tokens: int = 1024,
) -> str:
    """Send a multi-turn conversation to Claude and return the text response."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=messages,
    )
    for block in response.content:
        if block.type == "text":
            return block.text
    return ""


def reason_over_diseases(
    symptoms_text: str,
    all_diseases: list[dict],
    image_b64: Optional[str] = None,
    image_path: Optional[str] = None,
    media_type: str = "image/jpeg",
) -> str:
    """
    Unified Claude reasoning call.
    - Receives full symptom state + ALL 15 diseases (no pre-filtering).
    - Returns top 3 conditions with confidence scores + next diagnostic questions.
    """
    diseases_text = json.dumps(all_diseases, ensure_ascii=False, indent=2)

    prompt = f"""Sen temkinli ve kullanıcı odaklı çalışan bir dermatoloji bilgi asistanısın.

Görevin kesin teşhis koymak değildir.
Görevin, verilen semptom bilgileri ve varsa görüntü gözlemleri üzerinden,
sağlanan hastalık listesi içinden en olası dermatolojik durumları sıralamaktır.

KLİNİK YAKLAŞIM:
- Önce semptom örüntüsünü değerlendir.
- Lezyonun tipi, şekli, rengi, dağılımı, yerleşimi, süresi, büyüme hızı, kaşıntı/yanma/ağrı, akıntı, kanama, kabuklanma, kabarcık varlığı gibi ipuçlarını kullan.
- Görüntü varsa yalnızca görüntüde gerçekten seçilebilen özellikleri dikkate al; emin olmadığın görsel özellikleri varsayma.
- Verilen hastalık listesi dışına çıkma.
- Belirsizlik varsa bunu confidence değerine yansıt.
- Benign, inflamatuar veya enfeksiyöz açıklamalar güçlü ise; yalnızca yeterli bulgu olduğunda malign / yüksek riskli durumları üst sıralara taşı.
- Yüzde görülen akut inflamatuar, sivilce benzeri veya irinli lezyonlarda; güçlü malignite bulguları yoksa önce akne, rosacea, enfeksiyon veya benign inflamatuar nedenleri değerlendir.
- Maligniteyi ancak şu tip güçlü ipuçlarında üst sıralara taşı: uzun süreli iyileşmeyen lezyon, tekrarlayan kanama, belirgin ülserasyon, inci görünüm, düzensiz sınır, belirgin renk değişimi, hızlı ve açıklanamayan büyüme.
- Kullanıcıya panik yaratacak gereksiz kesin ifadeler kullanma.
- Bu çıktı bilgilendirme amaçlıdır; kesin tanı dili kullanma.

ÖNEMLİ SINIRLAR:
- Bu bir tıbbi teşhis değildir.
- Yalnızca verilen hastalık listesinden seçim yap.
- Hastalık isimlerini tıbbi olarak doğru kullan.
- Gerekçe kısa, net ve Türkçe olmalı.
- next_questions alanındaki sorular gerçekten ayırıcı tanıya yardımcı, hedefli ve kısa olmalı; öncelikle mevcut aday hastalıkları birbirinden ayıracak sorular sor (dağılım, süre, kaşıntı/yanma farkı, kabarcık varlığı, tek taraflılık, halka şekli, pullanma, kanama gibi).
- Gereksiz genel sorular sorma.
- Aynı bilgiyi tekrar eden soru sorma.
- Confidence değeri, olasılık sıralaması içindir; kesinlik anlamına gelmez.

SEMPTOM DURUMU:
{symptoms_text}

HASTALIK LİSTESİ:
{diseases_text}

GÖREV:
Verilen hastalık listesi içinden semptomlara en çok uyan 3 hastalığı seç.
Her biri için kısa bir Türkçe gerekçe yaz.
Tanıyı netleştirmeye en çok yardımcı olacak 2 veya 3 kısa hedefli soru üret.

ÇIKTI KURALLARI:
- En fazla 3 hastalık seç
- Yalnızca verilen listedeki hastalık adlarını kullan
- confidence: 0.0 ile 1.0 arasında sayı
- reason: Türkçe, kısa, 1-2 cümle
- next_questions: Türkçe, kısa, ayırıcı tanıya yardımcı 2-3 soru
- Ek açıklama yazma
- Markdown kullanma
- Yalnızca geçerli JSON döndür

JSON formatı:
{{
  "conditions": [
    {{
      "name": "...",
      "confidence": 0.82,
      "reason": "..."
    }},
    {{
      "name": "...",
      "confidence": 0.61,
      "reason": "..."
    }},
    {{
      "name": "...",
      "confidence": 0.37,
      "reason": "..."
    }}
  ],
  "next_questions": [
    "Soru 1?",
    "Soru 2?",
    "Soru 3?"
  ]
}}"""

    content: list[dict] = []

    if image_b64:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": image_b64},
        })
    elif image_path:
        path = Path(image_path)
        raw = path.read_bytes()
        b64 = base64.standard_b64encode(raw).decode("utf-8")
        suffix = path.suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
        mt = mime_map.get(suffix, "image/jpeg")
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": mt, "data": b64},
        })

    content.append({"type": "text", "text": prompt})

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": content}],
    )
    for block in response.content:
        if block.type == "text":
            return block.text
    return '{"conditions": [], "next_questions": []}'


def analyze_with_image(
    symptoms_text: str,
    candidate_diseases: list[dict],
    image_path: Optional[str] = None,
    image_b64: Optional[str] = None,
    media_type: str = "image/jpeg",
) -> str:
    """
    Ask Claude to reason over symptoms + candidate diseases,
    optionally including a skin image.
    Returns structured JSON as a string.
    """
    candidates_text = json.dumps(candidate_diseases, ensure_ascii=False, indent=2)

    reasoning_prompt = f"""Sen bir dermatoloji bilgi asistanısın.

Kullanıcının verdiği semptomlara göre olası dermatolojik durumları değerlendir.

ÖNEMLİ UYARI: Bu bir tıbbi teşhis değildir. Yalnızca bilgilendirme amaçlıdır.

Semptomlar:
{symptoms_text}

Aday hastalıklar:
{candidates_text}

En olası 3 hastalığı sırala ve kısa Türkçe açıklama ver.

MUTLAKA aşağıdaki JSON formatında yanıt ver (başka hiçbir şey ekleme):
{{
  "candidate_conditions": [
    {{
      "name": "Hastalık Adı",
      "reason": "Bu hastalığın neden olası olduğunu kısaca açıkla"
    }}
  ]
}}"""

    content: list[dict] = []

    if image_b64:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": image_b64,
            },
        })
    elif image_path:
        path = Path(image_path)
        raw = path.read_bytes()
        b64 = base64.standard_b64encode(raw).decode("utf-8")
        suffix = path.suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
        mt = mime_map.get(suffix, "image/jpeg")
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": mt, "data": b64},
        })

    content.append({"type": "text", "text": reasoning_prompt})

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": content}],
    )
    for block in response.content:
        if block.type == "text":
            return block.text
    return '{"candidate_conditions": []}'


def extract_slots_from_message(user_message: str) -> str:
    """
    Claude-based slot filling on a single user message.
    Handles Turkish morphology, synonyms, and informal text.
    Returns a JSON string to be merged into SymptomState.
    """
    prompt = f"""Aşağıdaki kullanıcı mesajından dermatolojik semptomları çıkar.

Kullanıcı mesajı:
"{user_message}"

Aşağıdaki alanları doldur. Bilgi yoksa null döndür.
Semptomları normalize et (ör: "büyüyor", "büyüdü", "büyümekte" → "büyüme").

YALNIZCA JSON döndür:
{{
  "location": "lezyonun bulunduğu bölge (ör: kol, yüz, sırt)",
  "symptoms": ["semptom1", "semptom2"],
  "growth": "evet veya hayır veya null",
  "duration": "ne zamandır var (ör: 2 haftadır, son zamanlarda)",
  "pain": "evet veya hayır veya null",
  "itching": "evet veya hayır veya null",
  "appearance": "görünüm açıklaması (ör: koyu renkli, düzensiz kenarlı)"
}}"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    for block in response.content:
        if block.type == "text":
            return block.text
    return "{}"


def predict_groups(symptoms_text: str, groups: list[dict]) -> list[int]:
    """
    Stage 1 of the two-stage diagnosis pipeline.
    Returns the top 2-3 group_ids most likely to contain the correct diagnosis.
    Falls back to [] on parse failure (caller should use all diseases).
    """
    import re
    groups_text = json.dumps(groups, ensure_ascii=False, indent=2)

    prompt = f"""Sen temkinli çalışan bir dermatoloji bilgi asistanısın.
Görevin kesin teşhis koymak değil, verilen semptomlardan yola çıkarak
en olası hastalık gruplarını belirlemektir.

KLİNİK YAKLAŞIM:
- Önce semptom örüntüsünü değerlendir.
- Lezyon tipi, dağılım, yerleşim, süre, kaşıntı/yanma/ağrı, büyüme, akıntı, kabarcık, renk değişimi gibi ipuçlarını dikkate al.
- Belirsizlik varsa en olası 2-3 grubu seç.
- Yalnızca güçlü kanıt varsa malign veya yüksek riskli grupları öne çıkar.
- İnflamatuar, enfeksiyöz veya benign açıklamalar güçlü görünüyorsa onları önce değerlendir.
- Bu aşamada hastalık adı değil, yalnızca grup seçimi yap.

SEMPTOM DURUMU:
{symptoms_text}

HASTALIK GRUPLARI:
{groups_text}

GÖREV:
Yukarıdaki semptom bilgilerine göre en olası 2 veya 3 hastalık grubunu seç.

ÇIKTI KURALLARI:
- Sadece verilen grup kimliklerinden seç
- En fazla 3 grup seç
- Açıklama yazma
- Ek metin yazma
- Yalnızca geçerli JSON döndür

JSON formatı:
{{"group_ids": [1, 5, 9]}}"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=128,
        messages=[{"role": "user", "content": prompt}],
    )
    for block in response.content:
        if block.type == "text":
            match = re.search(r"\{.*\}", block.text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                    ids = data.get("group_ids", [])
                    if isinstance(ids, list):
                        return [int(i) for i in ids]
                except (json.JSONDecodeError, ValueError):
                    pass
    return []


def extract_symptom_state(conversation_text: str) -> dict:
    """Use Claude to extract structured symptom slots from raw conversation."""
    system = """Sen bir dermatoloji asistanısın. Verilen konuşma metninden semptom bilgilerini çıkar.
Yalnızca JSON formatında yanıt ver."""

    prompt = f"""Aşağıdaki konuşmadan semptom bilgilerini çıkar:

{conversation_text}

Şu alanları doldur (bilgi yoksa null bırak):
{{
  "location": "lezyonun bulunduğu bölge",
  "symptoms": ["semptom1", "semptom2"],
  "duration": "ne zamandır var",
  "appearance": "nasıl görünüyor",
  "pain": "ağrı var mı",
  "itching": "kaşıntı var mı",
  "growth": "büyüme var mı"
}}"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    for block in response.content:
        if block.type == "text":
            return block.text
    return "{}"
