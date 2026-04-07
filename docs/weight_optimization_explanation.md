# Weight Optimizasyonu - Detaylı Açıklama

## Genel Mantık

Optimizasyon, her görüntü için GradCAM heatmap'inin ne kadarının ROI (göz/ağız) bölgelerinde yoğunlaştığını ölçer ve buna göre sample weight'leri belirler.

---

## ADIM 1: Focus Ratio Hesaplama

Her görüntü için GradCAM heatmap'i hesaplanır ve mask ile çarpılır:

```python
focus_ratio = sum(heatmap * mask) / sum(heatmap)
```

**Sayısal Örnek:**
- Görüntü A: Heatmap toplamı = 100, ROI'deki heatmap = 80 → Focus ratio = 0.80 ✅ (İyi)
- Görüntü B: Heatmap toplamı = 100, ROI'deki heatmap = 30 → Focus ratio = 0.30 ❌ (Kötü)
- Görüntü C: Heatmap toplamı = 100, ROI'deki heatmap = 95 → Focus ratio = 0.95 ✅✅ (Çok iyi)

**Örnek Dataset:**
```
Görüntüler: [A, B, C, D, E, F, G, H, I, J]
Focus ratios: [0.80, 0.30, 0.95, 0.65, 0.50, 0.75, 0.40, 0.85, 0.55, 0.70]
```

---

## ADIM 2: İstatistiksel Analiz

Focus ratio'ların dağılımı analiz edilir:

```python
mean = 0.645
median = 0.675
std = 0.198
Q25 = 0.50
Q75 = 0.80
```

**Yorum:**
- Ortalama 0.645 → Orta seviye
- Standart sapma 0.198 → Orta düzeyde dağınık
- Q25=0.50, Q75=0.80 → %50'si 0.50-0.80 arası

---

## ADIM 3: Target Focus Belirleme

Hedef focus ratio belirlenir. Amaç: Modelin ne kadar ROI odaklı olmasını istiyoruz?

```python
adaptive_offset = min(0.15, max(0.05, std * 0.5))
target_focus = min(0.92, max(0.45, median + adaptive_offset))
```

**Sayısal Örnek:**
- std = 0.198
- adaptive_offset = min(0.15, max(0.05, 0.198 * 0.5)) = min(0.15, max(0.05, 0.099)) = 0.099
- median = 0.675
- target_focus = min(0.92, max(0.45, 0.675 + 0.099)) = min(0.92, 0.774) = **0.774**

**Mantık:**
- Median'dan biraz yukarı hedef koyuyoruz (daha iyi performans için)
- Std yüksekse → daha küçük offset (konservatif)
- Std düşükse → daha büyük offset (agresif)

**Örnek Senaryolar:**
- Senaryo 1: median=0.5, std=0.3 (çok dağınık) → offset=0.15 → target=0.65
- Senaryo 2: median=0.7, std=0.1 (az dağınık) → offset=0.10 → target=0.80

---

## ADIM 4: Alpha Optimizasyonu (EN ÖNEMLİ ADIM)

Alpha, weight'lerin ne kadar agresif değişeceğini belirler. Farklı alpha değerleri test edilir.

### 4.1. Alpha Adayları Oluşturma

```python
alpha_min = 0.3
alpha_max = 3.0
candidates = [0.3, 0.49, 0.69, 0.89, 1.09, 1.29, 1.49, 1.69, 1.89, 2.09, 2.29, 2.49, 2.69, 2.89, 3.0]
```

### 4.2. Her Alpha İçin Penalty Hesaplama

**Penalty = Ortalama uzaklık × alpha**

```python
def simulate_penalty(alpha):
    penalties = [abs(target_focus - r) * alpha for r in ratios]
    return np.mean(penalties)
```

**Sayısal Örnek (target_focus = 0.774):**

Focus ratios: [0.80, 0.30, 0.95, 0.65, 0.50, 0.75, 0.40, 0.85, 0.55, 0.70]

**Alpha = 0.5 için:**
```
Uzaklıklar: [0.026, 0.474, 0.176, 0.124, 0.274, 0.024, 0.374, 0.076, 0.224, 0.074]
Penalties:  [0.013, 0.237, 0.088, 0.062, 0.137, 0.012, 0.187, 0.038, 0.112, 0.037]
Ortalama penalty = 0.082
```

**Alpha = 1.5 için:**
```
Uzaklıklar: [0.026, 0.474, 0.176, 0.124, 0.274, 0.024, 0.374, 0.076, 0.224, 0.074]
Penalties:  [0.039, 0.711, 0.264, 0.186, 0.411, 0.036, 0.561, 0.114, 0.336, 0.111]
Ortalama penalty = 0.247
```

**Alpha = 2.5 için:**
```
Uzaklıklar: [0.026, 0.474, 0.176, 0.124, 0.274, 0.024, 0.374, 0.076, 0.224, 0.074]
Penalties:  [0.065, 1.185, 0.440, 0.310, 0.685, 0.060, 0.935, 0.190, 0.560, 0.185]
Ortalama penalty = 0.412
```

### 4.3. En İyi Alpha Seçimi

**Tüm alpha'lar için penalty'ler:**
```
alpha=0.3  → penalty=0.049
alpha=0.5  → penalty=0.082
alpha=0.7  → penalty=0.115
alpha=0.9  → penalty=0.148
alpha=1.1  → penalty=0.181
alpha=1.3  → penalty=0.214
alpha=1.5  → penalty=0.247
alpha=1.7  → penalty=0.280
alpha=1.9  → penalty=0.313
alpha=2.1  → penalty=0.346
alpha=2.3  → penalty=0.379
alpha=2.5  → penalty=0.412
alpha=2.7  → penalty=0.445
alpha=2.9  → penalty=0.478
alpha=3.0  → penalty=0.494
```

**En düşük penalty = 0.049 → Best alpha = 0.3**

**Neden en düşük penalty seçiliyor?**
- Düşük penalty = focus ratio'lar target'a yakın
- Yüksek penalty = focus ratio'lar target'tan uzak
- En düşük penalty = en iyi ayrıştırma sağlayan alpha

**AMA DİKKAT:** Bu mantık biraz yanıltıcı! Düşük alpha her zaman düşük penalty verir çünkü penalty = uzaklık × alpha. 

**Gerçek optimizasyon mantığı:**
- Alpha'yı seçerken, weight'lerin ne kadar farklılaşacağına bakmalıyız
- Yüksek alpha → daha fazla farklılaşma → kötü örnekler daha fazla vurgulanır
- Düşük alpha → daha az farklılaşma → tüm örnekler benzer weight alır

**Daha iyi bir yaklaşım:** Weight dağılımının standart sapmasını maksimize etmek olabilir.

---

## ADIM 5: Clip Range Hesaplama

Weight'lerin min/max sınırları belirlenir:

```python
clip_range_factor = best_alpha * clip_range_factor_mult
clip_min = 1.0 - clip_range_factor
clip_max = 1.0 + clip_range_factor
```

**Sayısal Örnek (best_alpha = 0.3, clip_range_factor_mult = 0.8):**
- clip_range_factor = 0.3 * 0.8 = 0.24
- clip_min = 1.0 - 0.24 = 0.76
- clip_max = 1.0 + 0.24 = 1.24
- Final: clip_min = max(0.1, 0.76) = 0.76
- Final: clip_max = min(5.0, 1.24) = 1.24

**Sayısal Örnek (best_alpha = 2.0, clip_range_factor_mult = 0.8):**
- clip_range_factor = 2.0 * 0.8 = 1.6
- clip_min = 1.0 - 1.6 = -0.6 → max(0.1, -0.6) = 0.1
- clip_max = 1.0 + 1.6 = 2.6 → min(5.0, 2.6) = 2.6

---

## ADIM 6: Weight Uygulama

Her görüntü için weight hesaplanır:

### 6.1. Reward Mode (Şu anki ayar)

```python
delta = r - target_focus
w = 1 + alpha * delta
w = clip(w, clip_min, clip_max)
```

**Sayısal Örnek (target_focus=0.774, alpha=0.3, clip=[0.76, 1.24]):**

| Görüntü | Focus Ratio | Delta | Weight (1 + 0.3*delta) | Final Weight |
|---------|-------------|-------|------------------------|--------------|
| A       | 0.80        | +0.026| 1.008                  | 1.008 ✅     |
| B       | 0.30        | -0.474| 0.858                  | 0.858 ✅     |
| C       | 0.95        | +0.176| 1.053                  | 1.053 ✅     |
| D       | 0.65        | -0.124| 0.963                  | 0.963 ✅     |
| E       | 0.50        | -0.274| 0.918                  | 0.918 ✅     |
| F       | 0.75        | -0.024| 0.993                  | 0.993 ✅     |
| G       | 0.40        | -0.374| 0.888                  | 0.888 ✅     |
| H       | 0.85        | +0.076| 1.023                  | 1.023 ✅     |
| I       | 0.55        | -0.224| 0.933                  | 0.933 ✅     |
| J       | 0.70        | -0.074| 0.978                  | 0.978 ✅     |

**Yorum:**
- Yüksek focus ratio (C: 0.95) → Yüksek weight (1.053) → Ödüllendirilir
- Düşük focus ratio (B: 0.30) → Düşük weight (0.858) → Cezalandırılır

### 6.2. Penalize Mode (Alternatif)

```python
delta = target_focus - r
w = 1 + alpha * delta
w = clip(w, clip_min, clip_max)
```

**Aynı örnek için (target_focus=0.774, alpha=0.3, clip=[0.76, 1.24]):**

| Görüntü | Focus Ratio | Delta | Weight (1 + 0.3*delta) | Final Weight |
|---------|-------------|-------|------------------------|--------------|
| A       | 0.80        | -0.026| 0.992                  | 0.992 ✅     |
| B       | 0.30        | +0.474| 1.142                  | 1.142 ✅     |
| C       | 0.95        | -0.176| 0.947                  | 0.947 ✅     |
| D       | 0.65        | +0.124| 1.037                  | 1.037 ✅     |
| E       | 0.50        | +0.274| 1.082                  | 1.082 ✅     |
| F       | 0.75        | +0.024| 1.007                  | 1.007 ✅     |
| G       | 0.40        | +0.374| 1.112                  | 1.112 ✅     |
| H       | 0.85        | -0.076| 0.977                  | 0.977 ✅     |
| I       | 0.55        | +0.224| 1.067                  | 1.067 ✅     |
| J       | 0.70        | +0.074| 1.022                  | 1.022 ✅     |

**Yorum:**
- Düşük focus ratio (B: 0.30) → Yüksek weight (1.142) → Cezalandırılır (daha fazla öğrenir)
- Yüksek focus ratio (C: 0.95) → Düşük weight (0.947) → Ödüllendirilir (az öğrenir)

---

## Özet: Optimizasyon Kararları

### 1. Target Focus
- **Ne zaman:** Focus ratio dağılımına göre
- **Nasıl:** Median + adaptive offset
- **Neden:** Modelin ne kadar ROI odaklı olmasını istediğimizi belirler

### 2. Alpha
- **Ne zaman:** Tüm alpha adayları test edilir
- **Nasıl:** Her alpha için penalty hesaplanır, en düşük seçilir
- **Neden:** Weight'lerin ne kadar farklılaşacağını belirler
- **Sorun:** Mevcut yaklaşım her zaman düşük alpha seçer (penalty = uzaklık × alpha)

### 3. Clip Range
- **Ne zaman:** Alpha belirlendikten sonra
- **Nasıl:** `alpha × clip_range_factor_mult` ile hesaplanır
- **Neden:** Aşırı weight'leri önlemek için

### 4. Weight Uygulama
- **Ne zaman:** Her görüntü için
- **Nasıl:** `w = 1 + alpha × delta`
- **Neden:** Focus ratio'ya göre örnek önemini belirler

---

## Mevcut Optimizasyonun Sorunları

1. **Alpha seçimi yanıltıcı:** Düşük alpha her zaman düşük penalty verir
2. **Penalty formülü eksik:** Sadece ortalama uzaklık kullanılıyor
3. **Weight dağılımı kontrol edilmiyor:** Alpha seçilirken weight'lerin gerçek dağılımına bakılmıyor

---

## Önerilen İyileştirmeler

1. **Weight dağılımı standart sapmasını maksimize et:**
   ```python
   def evaluate_alpha(alpha):
       weights = [1 + alpha * (r - target_focus) for r in ratios]
       weights = [clip(w, clip_min, clip_max) for w in weights]
       return np.std(weights)  # Daha yüksek = daha iyi ayrıştırma
   ```

2. **Weight range'ini maksimize et:**
   ```python
   def evaluate_alpha(alpha):
       weights = [1 + alpha * (r - target_focus) for r in ratios]
       weights = [clip(w, clip_min, clip_max) for w in weights]
       return max(weights) - min(weights)  # Daha geniş = daha iyi
   ```

3. **Kötü örneklerin weight'ini maksimize et:**
   ```python
   def evaluate_alpha(alpha):
       weights = [1 + alpha * (r - target_focus) for r in ratios]
       weights = [clip(w, clip_min, clip_max) for w in weights]
       bad_samples = [w for r, w in zip(ratios, weights) if r < target_focus]
       return np.mean(bad_samples)  # Kötü örnekler daha fazla vurgulanır
   ```

