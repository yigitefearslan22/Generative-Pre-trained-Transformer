# GPT Language Model — From Scratch with PyTorch

Sıfırdan PyTorch ile yazılmış, Andrej Karpathy'nin GPT mimarisinden ilham alan bir **Transformer tabanlı dil modeli** projesidir. Model, OpenWebText veri seti üzerinde pre-train edilmiş ve ardından Stanford Alpaca (instruction-tuning) veri seti ile fine-tune edilmiştir.

> **⚠️ Bu proje herhangi bir yapay zeka asistanı (ChatGPT, Copilot, vb.) kullanılmadan, tamamen elle yazılmıştır.** Tüm model mimarisi, eğitim döngüsü, veri işleme pipeline'ı ve fine-tuning kodu sıfırdan, Andrej Karpathy'nin eğitim videolarından öğrenilerek geliştirilmiştir. Kodlardaki Türkçe yorumlar ve değişken isimleri de bu öğrenme sürecinin bir parçasıdır.

> **Not:** Bu model bir bigram modeli **değildir**. Multi-Head Self-Attention, FeedForward katmanları, LayerNorm ve residual bağlantılar içeren tam bir GPT-tarzı Transformer mimarisine sahiptir.

---

## 📐 Model Mimarisi

| Parametre | Değer |
|---|---|
| Embedding boyutu (`n_embd`) | 384 |
| Attention head sayısı (`n_head`) | 8 |
| Transformer blok sayısı (`n_layer`) | 8 |
| Context uzunluğu (`block_size`) | 64 token |
| Dropout oranı | 0.2 |
| Tokenizer | GPT-2 (tiktoken, ~50k vocab) |
| Optimizer | AdamW (openwebtext traini için lr = 3e-4)(alpaca fine-tune için lr = 2e-4) |
| LR Scheduling | Cosine Annealing (eta_min = 1e-6) |
| Gradient Clipping | max_norm = 1.0 |
| Sampling | Temperature (0.8) + Top-k (50) |

### Mimari Bileşenler

- **`Head`** — Tek bir causal self-attention head'i (key, query, value projeksiyonları + causal mask)
- **`MultiHeadAttention`** — Birden fazla Head'i paralel çalıştırır, sonuçları birleştirir + residual projection scaling
- **`FeedForward`** — İki katmanlı MLP (4x genişleme + GELU + dropout) + residual projection scaling
- **`Block`** — Pre-LayerNorm Transformer bloğu (Self-Attention → FeedForward, residual bağlantılarla)
- **`GPTLanguageModel`** — Token + pozisyon embedding (weight tying), N adet Block, son LayerNorm ve linear head

---

## 📁 Dosya Yapısı

```
fcc-gpt-course/
├── gpt_language_model.ipynb   # Ana model notebook'u (eğitim, fine-tuning, inference)
├── torch-examples.ipynb       # PyTorch temel işlemleri ve öğrenme notları
├── download_alpaca_data.ipynb  # Alpaca veri setini indirmek için yardımcı notebook
├── requirements.txt           # Python bağımlılıkları
├── wizard_of_oz.txt           # Orijinal pre-training metin verisi (opsiyonel)
├── alpaca_data.json           # Stanford Alpaca instruction-tuning veri seti (~52k örnek)
├── ai_save.pt                 # Pre-trained model ağırlıkları (OpenWebText)
├── ai_save_alpaca.pt          # Fine-tuned model ağırlıkları (Alpaca)
├── cuda/                      # Python sanal ortamı (CUDA destekli)
└── README.md                  # Bu dosya
```

---

## 🛠️ Kurulum

### Gereksinimler

- Python 3.12+
- NVIDIA GPU + CUDA (önerilir, CPU'da da çalışır ama çok yavaştır)
- Jupyter Notebook veya JupyterLab

### 1. Sanal Ortamı Kullanma

Projede zaten bir sanal ortam (`cuda/`) mevcut. Aktifleştirmek için:

```bash
# Windows
.\cuda\Scripts\activate

# Linux / macOS
source cuda/bin/activate
```

### 2. Gerekli Paketleri Yükleme

```bash
# CUDA destekli PyTorch kurulumu (GPU kullanacaklar için)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Diğer tüm bağımlılıklar
pip install -r requirements.txt
```

`requirements.txt` içeriği:
| Paket | Açıklama |
|---|---|
| `torch` | PyTorch — model eğitimi ve CUDA desteği |
| `tiktoken` | OpenAI'ın GPT-2 tokenizer'ı |
| `datasets` | HuggingFace Datasets — OpenWebText yükleme |
| `numpy` | Sayısal işlemler ve benchmark |
| `jupyter` | Notebook çalıştırma ortamı |

### 3. Alpaca Veri Setini İndirme

`download_alpaca_data.ipynb` notebook'unu çalıştırarak Alpaca veri setini indirebilirsiniz, ya da elle:

```bash
curl -o alpaca_data.json https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json
```

---

## 🚀 Çalıştırma

### Jupyter Notebook'u Başlatma

```bash
jupyter notebook gpt_language_model.ipynb
```

### Notebook Hücre Sırası ve Açıklamaları

`gpt_language_model.ipynb` aşağıdaki sırayla çalıştırılmalıdır:

#### Aşama 1 — Hiperparametrelerin Tanımlanması (Hücre 2)
CUDA cihaz kontrolü, `block_size`, `batch_size`, `n_embd`, `n_head`, `n_layer` gibi tüm hiperparametreler burada ayarlanır. **Her zaman ilk çalıştırılması gereken hücredir.**

#### Aşama 2 — Veri Hazırlığı (İki seçenek)

**Seçenek A: OpenWebText ile Pre-Training (Hücre 1 → Hücre 3 → Hücre 4)**
1. **Hücre 1:** HuggingFace'den OpenWebText veri setini yükler (ilk seferde uzun sürer)
2. **Hücre 3:** Tiktoken (GPT-2) tokenizer ile metni tokenize eder, ilk 10.000 kaydı kullanır
3. **Hücre 4:** Veriyi %90 train / %10 validation olarak böler

**Seçenek B: Alpaca ile Fine-Tuning (Hücre 5)**
- `alpaca_data.json` dosyasından instruction-response formatında veri yükler
- `### Talimat: ... ### Cevap: ...` formatında prompt'lar oluşturur
- Otomatik olarak train/val split yapar

#### Aşama 3 — Yardımcı Fonksiyonlar (Hücre 5–6)
- **`get_batch()`**: Rastgele mini-batch'ler oluşturur
- **`estimate_loss()`**: Gradient hesaplanmadan ortalama train ve val loss hesaplar

#### Aşama 4 — Model Tanımı (Hücre 7)
Tüm model sınıfları burada tanımlıdır: `Head`, `MultiHeadAttention`, `FeedForward`, `Block`, `GPTLanguageModel`. **Bu hücre mutlaka model yüklenmeden veya eğitim başlamadan önce çalıştırılmalıdır.**

#### Aşama 5 — Model Yükleme veya Eğitim

**Var olan ağırlıkları yükleyerek başlama:**
- **Hücre 8:** `ai_save.pt` dosyasından pre-trained ağırlıkları yükler
- **Hücre 9:** `ai_save_alpaca.pt` dosyasından fine-tuned ağırlıkları yükler

**Sıfırdan veya devam ederek eğitim:**
- **Hücre 10:** Eğitim döngüsü. AdamW optimizer + Cosine Annealing LR scheduler ile `max_iters` adım eğitir. Gradient clipping (max_norm=1.0) ile eğitim stabilitesi sağlanır. Her `eval_iters` adımda loss ve learning rate yazdırır, model kaydeder.

#### Aşama 6 — Metin Üretimi (Inference)

- **Hücre 11 — Serbest üretim:** Boş bir context'ten başlayarak temperature=0.8 ve top-k=50 sampling ile 500 token üretir
- **Hücre 12 — Sohbet modu:** Bir talimat (instruction) verip modelden cevap alır. Temperature/top-k sampling ve EOS token durdurma ile `### Talimat:` / `### Cevap:` formatını kullanır

---

## 💡 Hızlı Başlangıç: Sadece Inference (Eğitim Yapmadan)

Eğer sadece modeli test etmek istiyorsanız, aşağıdaki hücreleri sırayla çalıştırın:

1. **Hücre 2** — Hiperparametreleri tanımla
2. **Hücre 7** — Model sınıflarını tanımla
3. **Hücre 9** — Fine-tuned ağırlıkları yükle (`ai_save_alpaca.pt`)
4. **Hücre 12** — Sohbet modunda soru sor

---

## ⚠️ Bilinen Sınırlamalar

- Model küçük ölçekli bir GPT'dir (~10M parametre civarı); çıktılar genellikle tutarsız ve düşük kaliteli olacaktır
- Context uzunluğu 64 token ile sınırlıdır — uzun bağlam gerektiren sorularda başarısız olur
- Alpaca fine-tuning az iterasyonla yapıldığından instruction-following kapasitesi sınırlıdır
- `.pt` ağırlık dosyaları ~200MB boyutundadır; Git LFS kullanmanız önerilir

## 🔧 Daha İyi Sonuçlar İçin Öneriler

Mevcut model düşük hiperparametrelerle eğitilmiştir. Daha tutarlı ve kaliteli çıktılar almak istiyorsanız, **`block_size` ve `batch_size` değerlerini arttırarak modeli sıfırdan yeniden pre-train etmeniz** gerekir.

| Parametre | Mevcut Değer | Önerilen Değer | Açıklama |
|---|---|---|---|
| `block_size` | 64 | 256–512 | Modelin bir seferde gördüğü token sayısı. Arttırıldığında model daha uzun bağlamları öğrenir ve daha tutarlı cümleler üretir. |
| `batch_size` | 8 | 32–128 | Paralel işlenen örnek sayısı. Arttırıldığında gradient tahminleri daha kararlı olur ve eğitim daha stabil ilerler. |
| `max_iters` | 500 | 3000–10000+ | Eğitim adım sayısı. Daha büyük veri ve model için daha fazla iterasyon gerekir. |
| `n_layer` | 8 | 8–12 | Transformer blok sayısı. VRAM yeterliyse arttırılabilir. |

> **⚠️ Önemli:** `block_size` ve `batch_size` arttırıldığında GPU bellek (VRAM) kullanımı önemli ölçüde artar. En az **8GB VRAM** (RTX 3060/4060 ve üzeri) önerilir. VRAM yetersizse `batch_size`'ı düşük tutup `block_size`'ı kademeli olarak arttırın.
>

---

## 📚 Ek Notebook'lar

### `torch-examples.ipynb`
PyTorch temel kavramlarını öğrenmek için kullanılan bir alıştırma notebook'u:
- Tensor oluşturma (`zeros`, `ones`, `rand`, `eye`, `arange`, `linspace`, `logspace`)
- GPU vs CPU performans karşılaştırması (matris çarpımı benchmark)
- `torch.multinomial`, `F.softmax`, `nn.Embedding` kullanım örnekleri

### `download_alpaca_data.ipynb`
Stanford Alpaca veri setini (`alpaca_data.json`) GitHub'dan indiren tek hücreli yardımcı notebook.

---

## 📖 Referanslar

- [Andrej Karpathy — Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [freeCodeCamp GPT Course](https://www.youtube.com/watch?v=UU1WVnMk4E8)
- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- [OpenWebText Dataset](https://huggingface.co/datasets/Skylion007/openwebtext)
