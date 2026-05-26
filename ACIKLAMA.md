# GPT Dil Modeli ve Derin Öğrenme Mekanizmaları Rehberi

Bu rehber, projedeki GPT dil modelinin mimarisini, PyTorch kütüphanesinin rolünü, kullanılan matematiksel optimizasyon algoritmalarını (özellikle **AdamW**) ve Büyük Dil Modellerinin (LLM) temel çalışma prensiplerini detaylandırmak amacıyla hazırlanmıştır.

---

## 1. Büyük Dil Modelleri (LLM) ve GPT Nedir?

### Büyük Dil Modelleri (LLM)
Büyük Dil Modelleri, kendilerine verilen bir metin bağlamını (context) kullanarak bir sonraki kelimeyi (veya token'ı) tahmin etmek üzere eğitilmiş devasa yapay sinir ağlarıdır. LLM'lerin temel görevi basittir: **Bir sonraki token'ın olasılık dağılımını hesaplamak.**

$$P(x_t \mid x_{1}, x_{2}, \dots, x_{t-1})$$

### GPT (Generative Pre-trained Transformer)
GPT, OpenAI tarafından geliştirilen ve sadece **Decoder (Kod Çözücü)** bloklarından oluşan bir Transformer mimarisidir. 
* **Generative (Üretken):** Yeni metinler üretebilir.
* **Pre-trained (Ön Eğitilmiş):** Büyük miktarda genel internet metni (örn. OpenWebText) üzerinde bir sonraki kelimeyi tahmin etme (Causal Language Modeling) yöntemiyle eğitilmiştir.
* **Transformer:** Mekansal bağımlılıkları ve uzun vadeli ilişkileri yakalamak için self-attention (öz-dikkat) mekanizmasını kullanan mimaridir.

---

## 2. Model Mimarisi ve Bileşenlerin Matematiksel Analizi

Modelimiz, standart bir **Decoder-only Transformer** olup şu katmanlardan oluşur:

### A. Token ve Positional Embedding (Ağırlık Bağlama ile)
1. **Token Embedding (`nn.Embedding`):** Her kelime/token, yüksek boyutlu sürekli bir vektör uzayına (`n_embd = 384`) eşlenir.
2. **Positional Embedding (`nn.Embedding`):** Transformer'lar kelimelerin sırasını algılayamaz (çünkü tüm token'ları paralel işler). Bu yüzden her token vektörüne, dizideki konumunu belirten pozisyon vektörleri eklenir.
3. **Weight Tying (Ağırlık Bağlama):** Girişteki Token Embedding matrisi ile en sondaki çıktı olasılıklarını hesaplayan Linear katmanının (Language Model Head) ağırlıkları birbirine eşitlenir (bağlanır). Bu teknik:
   * Model parametre sayısını ciddi oranda azaltır.
   * Aşırı öğrenmeyi (overfitting) engeller.
   * Modelin kelime temsil yeteneğini artırır.

```python
# Ağırlık Bağlama (Weight Tying) Örneği
self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
self.lm_head = nn.Linear(n_embd, vocab_size)
# Ağırlıkları eşitleme
self.lm_head.weight = self.token_embedding_table.weight
```

---

### B. Causal Self-Attention (Nedensel Öz-Dikkat)
Transformer mimarisinin kalbi **Attention** mekanizmasıdır. Causal (Nedensel) olması, bir token'ın yalnızca kendinden önceki token'lara dikkat edebileceği, gelecekteki token'ları göremeyeceği anlamına gelir. Bu işlem üst üçgensel bir maskeleme matrisi (`tril`) ile sağlanır.

Her bir dikkat kafasında (Head) girdiden üç temel vektör üretilir:
* **Query ($Q$ - Sorgu):** "Ne arıyorum?"
* **Key ($K$ - Anahtar):** "Bende ne var?"
* **Value ($V$ - Değer):** "Ne sunuyorum?"

Dikkat katsayıları şu formülle hesaplanır:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

* $\sqrt{d_k}$ terimi (ölçekleme), matris çarpımı büyüdüğünde softmax fonksiyonunun türevlerinin sıfıra yaklaşmasını (vanishing gradients) önlemek için kullanılır.

#### Multi-Head Attention (Çok Kafalı Dikkat)
Tek bir dikkat kafası yerine, modeli paralel olarak farklı temsil uzaylarında çalışan 8 adet kafa (`n_head = 8`) ile donatırız. Kafaların çıktıları birleştirilir (concatenation) ve doğrusal bir katmandan geçirilerek bir sonraki aşamaya aktarılır.

---

### C. FeedForward Ağları ve GELU Aktivasyon Fonksiyonu
Her dikkat bloğundan sonra token temsilini doğrusal olmayan şekilde işleyen iki katmanlı bir MLP (Multi-Layer Perceptron) bulunur. 

* **GELU (Gaussian Error Linear Unit):** Standart ReLU yerine modern Transformer'larda (GPT, BERT) GELU tercih edilir. 
  $$\text{GELU}(x) = x \Phi(x)$$
  (burada $\Phi(x)$ standart normal dağılımın kümülatif dağılım fonksiyonudur). GELU, negatif değerler için tamamen sıfır olmak yerine küçük negatif değerlerin geçmesine izin vererek ağın gradyan akışını daha sağlıklı tutar.

---

### D. Katman Normu (LayerNorm) ve Pre-LN Mimarisi
Eğitimin stabil olması ve derin ağların eğitilebilmesi için normalizasyon şarttır.
* **Pre-LN Mimarisi:** Katman normalizasyonu, Causal Self-Attention veya FeedForward katmanlarına **girmeden hemen önce** uygulanır. Orijinal Transformer makalesindeki Post-LN'ye göre Pre-LN, derin modellerin kararlı eğitilmesinde çok daha etkilidir.

$$\text{Pre-LN Layout: } x \rightarrow \text{LayerNorm}(x) \rightarrow \text{Attention} \rightarrow \text{Residual Add} \rightarrow \text{LayerNorm} \rightarrow \text{FeedForward} \rightarrow \text{Residual Add}$$

---

### E. Artık Bağlantılar (Residual Connections)
Her blok, girdisini doğrudan çıktıya ekleyen kısa yollar içerir: $y = x + f(x)$. Bu sayede geriye yayılım (backpropagation) esnasında gradyanlar hiçbir engele takılmadan derin katmanlardan giriş katmanlarına kadar akar ve gradyan yok olması (vanishing gradient) engellenir.

---

## 3. PyTorch'un Rolü ve CUDA Teknolojisi

**PyTorch**, dinamik hesaplama grafikleri (Dynamic Computation Graphs) sunan, Python tabanlı bir tensör kütüphanesidir.

### Tensörler ve Otomatik Gradyan (`Autograd`)
Modelin eğitilebilmesi için milyonlarca parametreye göre kaybın (loss) kısmi türevlerinin alınması gerekir. PyTorch, oluşturulan hesaplama grafiği üzerinden geriye doğru hareket ederek zincir kuralı (Chain Rule) ile tüm türevleri otomatik olarak hesaplar (`loss.backward()`).

### CUDA ve GPU İvmelendirmesi
Transformer modelleri, matris çarpımlarına dayalı yoğun paralel işlemlere ihtiyaç duyar.
* **CPU:** Az sayıda çekirdeğe sahiptir ve işlemleri sıralı yapmakta iyidir.
* **GPU (CUDA):** Binlerce küçük çekirdek içerir. Matris çarpımları gibi devasa paralel işlemleri CPU'ya kıyasla **100 ila 1000 kat daha hızlı** yapabilir.
* Kodumuzdaki `device = 'cuda' if torch.cuda.is_available() else 'cpu'` ifadesi, VRAM'de yer alan ekran kartı çekirdeklerini otomatik tespit edip modeli ve verileri oraya taşır (`.to(device)`).

---

## 4. Eğitim Dinamikleri ve Matematiksel Optimizasyon

### A. AdamW Optimizasyon Algoritması
Geleneksel Stokastik Gradyan İnişi (SGD) yerine projemizde **AdamW** optimizasyon algoritması kullanılmıştır.

#### Adam Algoritmasının Problemi:
Standart Adam algoritması, gradyanların birinci (ortalama) ve ikinci (varyans) momentlerini kullanarak parametre güncellemelerini ölçeklendirir (adaptif öğrenme oranı). Ancak üzerine L2 Regularizasyonu (Ağırlık Azaltımı - Weight Decay) eklendiğinde, gradyan güncellemesine ağırlıkların büyüklüğü dahil olur ve adaptif katsayı bu azaltımı bozar. Bu durum ağırlık azaltımının etkisini zayıflatır.

#### AdamW Çözümü (Decoupled Weight Decay):
Loshchilov ve Hutter (2017) tarafından önerilen **AdamW**, ağırlık azaltımını doğrudan gradyanlardan **ayırır** ve güncelleme adımının en sonunda doğrudan parametrelere uygular:

$$\theta_{t+1} = \theta_t - \eta_t \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$

* $\theta_t$: Model parametreleri (ağırlıklar)
* $\hat{m}_t$: Birinci moment tahmini (hızlanma/momentum)
* $\hat{v}_t$: İkinci moment tahmini (adaptif öğrenme hızı)
* $\lambda$: Weight Decay katsayısı (Ağırlık Azaltma oranı)
* $\eta_t$: Öğrenme oranı (Learning Rate)

Bu sayede model aşırı öğrenmeye (overfitting) karşı çok daha dirençli hale gelir ve ezberlemek yerine genel kalıpları öğrenir.

---

### B. Cosine Annealing Learning Rate Scheduler (Kosinüs Tavlama)
Sabit bir öğrenme oranıyla eğitim yapmak verimsizdir. Başlangıçta büyük adımlarla hızlı öğrenmek, sona doğru ise yerel minimumları kaçırmamak için adımları küçültmek gerekir.

**Cosine Annealing**, öğrenme oranını bir kosinüs dalgası şeklinde kademeli olarak düşürür:

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)$$

Bu sayede eğitim kararlı ilerler ve kayıp fonksiyonunun dip noktalarına daha yumuşak yerleşilir.

---

### C. Gradient Clipping (Gradyan Kırpma)
Derin Transformer yapılarında, bazı eğitim adımlarında gradyanlar aşırı derecede büyüyebilir. Bu duruma **gradyan patlaması (exploding gradients)** denir ve modelin tüm parametrelerini bozup eğitimi çökertebilir (`NaN loss`).

* **Çözüm:** Gradyan vektörünün L2 normu, belirlenen bir maksimum sınıra (`max_norm = 1.0`) kırpılır. Eğer norm bu değeri aşarsa doğrusal olarak küçültülür.

```python
# PyTorch ile Gradyan Kırpma
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

### D. Model Kaydetme ve Yükleme Yapılandırması (Save/Load Paths)
Eğitim döngüsünde model ağırlıklarının hangi adla kaydedileceğini belirlemek için notebook'un ikinci hücresine (hiperparametreler) dinamik bir `save_path` parametresi eklenmiştir:

* **Pre-training (Ön Eğitim) Aşaması:** Sıfırdan genel dil eğitimini yapıyorsanız, `save_path = 'ai_save.pt'` olarak ayarlamanız gerekir. Bu sayede modelin temel ağırlıkları kaydedilir.
* **Fine-tuning (İnce Ayar) Aşaması:** Alpaca gibi talimat/sohbet veri setleriyle çalışıyorsanız, `save_path = 'ai_save_alpaca.pt'` olarak ayarlamalızınız. 

Bu yapı sayesinde, genel dil yeteneklerine sahip temel modelinizin ağırlıkları (`ai_save.pt`) ezilmeden korunur ve asistan yetenekli modeliniz (`ai_save_alpaca.pt`) bağımsız olarak kaydedilir.

---

## 5. Çıkarım (Inference) ve Örnekleme Algoritmaları

Model eğitildikten sonra, çıktı üretirken ham olasılıkları (logits) doğrudan seçmek yerine yaratıcılığı ve kaliteyi dengeleyen teknikler kullanırız:

### A. Sıcaklık Parametresi (Temperature)
Softmax işlemine girmeden önce logits değerleri bir $T$ sıcaklık değerine bölünür:

$$P_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

* **Düşük Sıcaklık ($T < 1.0$):** Olasılık dağılımını daha dik ve belirgin hale getirir. Model daha kararlı, mantıklı ama kendini tekrarlayan çıktılar üretir.
* **Yüksek Sıcaklık ($T > 1.0$):** Dağılımı düzleştirir. Düşük olasılıklı kelimelerin seçilme şansı artar, model daha yaratıcı ama saçmalamaya yatkın çıktılar üretir (Projemizde dengeli bir değer olan `0.8` tercih edilmiştir).

### B. Top-K Örnekleme (Top-K Sampling)
Her adımda tüm sözlükteki (~50.000 kelime) olasılıkları değerlendirmek yerine, en yüksek olasılığa sahip ilk $K$ kelime (`top_k = 50`) filtrelenir ve diğerlerinin olasılığı sıfırlanır. Olasılık dağılımı bu 50 kelime arasında yeniden normalize edilerek örnekleme yapılır. Bu işlem modelin anlamsız ve alakasız kelimeler seçmesini engeller.

### C. EOS (End of Sequence) ile Durdurma
Özellikle sohbet (chatbot) modunda modelin sonsuza kadar anlamsız metinler üretmesini önlemek için, modelin ürettiği özel bitiş token'ı (EOS) yakalandığı anda metin üretimi otomatik olarak durdurulur.

---

## 6. Proje Veri Pipeline'ı

Model iki aşamalı bir veri akışıyla çalışır:

1. **Pre-Training (Ön Eğitim):** modelin dilin genel yapısını, dil bilgisini ve temel mantığı kavraması için **OpenWebText** veri setiyle bir sonraki kelime tahmini üzerine eğitilir.
2. **Instruction Fine-Tuning (Talimat İnce Ayarı):** Ön eğitimli model, kullanıcının komutlarına ve sorularına düzgün cevap verebilmesi amacıyla **Stanford Alpaca** veri setiyle (`### Talimat:` / `### Cevap:` yapısında) ince ayara tabi tutulur. Bu aşama modele bir "yardımcı asistan" kimliği kazandırır.
