# Eye-Mouth Focused Mask Integration

Bu dokümantasyon, drowsiness detection projesine göz ve ağız odaklı maske entegrasyonunu açıklar.

## 🎯 Amaç

Modelin sadece göz ve ağız bölgelerinden öğrenmesini sağlayarak daha etkili drowsiness detection elde etmek.

## 📁 Yeni Dosyalar

### `src/losses.py`
- **EyeMouthMaskGenerator**: Göz ve ağız bölgeleri için maske üretir
- **create_simple_masked_loss**: Sample weight ile çalışan loss fonksiyonu
- **EyeMouthGradientLoss**: Gradient-based loss sınıfı

### `test_mask_integration.py`
- Mask entegrasyonunu test eden script

## 🔧 Yapılan Değişiklikler

### 1. `src/dataloader.py`
```python
# Yeni parametre eklendi
def get_binary_pipelines(..., use_masks=False):
    # Mask generator entegrasyonu
    if use_masks:
        mask_generator = EyeMouthMaskGenerator(img_size)
        # Dataset'e mask ekleme
```

### 2. `src/train.py`
```python
# Yeni parametreler eklendi
def train_model(..., use_gradient_loss=False, lambda_grad=0.1, target_layer_name=None):
    # Loss fonksiyonu seçimi
    if use_gradient_loss:
        loss_fn = EyeMouthGradientLoss(...)
    else:
        loss_fn = create_simple_masked_loss(...)
```

### 3. `main.py`
```python
# Mask'ları etkinleştirme
train_ds, val_ds, test_ds, class_names = get_binary_pipelines(
    ..., use_masks=True  # Mask'ları etkinleştir
)

# Gradient loss parametreleri
history = train_model(
    ..., 
    use_gradient_loss=False,  # Sample weight yaklaşımı
    lambda_grad=0.1,          # Gradient penalty ağırlığı
    target_layer_name="conv2d_2"  # Hedef katman
)
```

## 🎨 Mask Tasarımı

### Göz Bölgesi
- **Konum**: Görüntünün üst %25-45'i
- **Genişlik**: Görüntünün %60'ı (merkez)
- **Amaç**: Göz kapanması, göz kırpma tespiti

### Ağız Bölgesi  
- **Konum**: Görüntünün %55-75'i
- **Genişlik**: Görüntünün %40'ı (merkez)
- **Amaç**: Esneme, ağız açıklığı tespiti

### Mask Değerleri
- **1.0**: Göz/ağız bölgesi (önemli)
- **0.0**: Diğer bölgeler (önemsiz)
- **Gaussian blur**: Yumuşak geçişler için

## 🚀 Kullanım

### Seçenek 1: Sample Weight Yaklaşımı (Önerilen)
```python
# Mask'ları etkinleştir
train_ds, val_ds, test_ds, class_names = get_binary_pipelines(
    data_dir, use_masks=True
)

# Sample weight ile eğitim
history = train_model(
    model, train_ds, val_ds,
    use_gradient_loss=False,  # Sample weight kullan
    lambda_grad=0.1
)
```

### Seçenek 2: Pixel-based Gradient Loss (Deneysel)
```python
# Mask'ları etkinleştir
train_ds, val_ds, test_ds, class_names = get_binary_pipelines(
    data_dir, use_masks=True
)

# Gradient-based loss ile eğitim
history = train_model(
    model, train_ds, val_ds,
    use_gradient_loss=True,  # Gradient loss kullan
    lambda_grad=0.1,
    target_layer_name="conv2d_2"
)
```

### main.py'de Seçim Yapma
```python
# main.py'de bu değişkenleri değiştirin:

# Seçenek 1: Sample Weight (Önerilen)
use_sample_weight = True
use_gradient_loss = False

# Seçenek 2: Gradient Loss (Deneysel)
use_sample_weight = False
use_gradient_loss = True
```

## ⚙️ Parametreler

### `lambda_grad`
- **Açıklama**: Gradient penalty ağırlığı
- **Önerilen değer**: 0.1
- **Yüksek değer**: Daha güçlü mask zorlaması
- **Düşük değer**: Daha esnek öğrenme

### `target_layer_name`
- **Açıklama**: Gradient hesaplama için hedef katman
- **Önerilen**: "conv2d_2" (ikinci conv katmanı)
- **Alternatifler**: "conv2d", "conv2d_1", "conv2d_3"

## 🔍 Test Etme

```bash
# Test scriptini çalıştır
python test_mask_integration.py

# Ana eğitimi çalıştır
python main.py
```

## 📊 Beklenen Faydalar

1. **Odaklanmış Öğrenme**: Model sadece önemli bölgelerden öğrenir
2. **Gürültü Azaltma**: Arka plan gürültüsü etkisini azaltır
3. **Hızlı Yakınsama**: Daha hedefli öğrenme
4. **Daha İyi Genelleme**: Sadece ilgili özellikler öğrenilir

## ⚠️ Dikkat Edilecekler

1. **Mask Kalitesi**: Mask'lar geometrik tahminlere dayanır
2. **Face Detection**: İleride gerçek yüz tespiti eklenebilir
3. **Performans**: Mask üretimi ek hesaplama yükü
4. **Hyperparameter Tuning**: `lambda_grad` değeri optimize edilmeli

## 🔮 Gelecek Geliştirmeler

1. **Gerçek Face Detection**: OpenCV/MediaPipe ile yüz tespiti
2. **Adaptive Masking**: Öğrenme sürecinde mask güncelleme
3. **Multi-Scale Masks**: Farklı çözünürlüklerde mask'lar
4. **Attention Visualization**: Mask'ların görselleştirilmesi

## 📝 Notlar

- Mevcut implementasyon **sample_weight** yaklaşımını kullanır
- **Gradient-based loss** deneysel aşamadadır
- Mask'lar her batch için dinamik olarak üretilir
- Tüm metrikler (accuracy, precision, recall, AUC) mask'ları otomatik kullanır
