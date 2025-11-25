# 🎨 Sketch-to-Code AI

> **Computer Vision** + **Generative AI** = Profesyonel Web Siteleri

El çizimi web sitesi taslağını profesyonel HTML/CSS koduna dönüştüren gelişmiş AI uygulaması.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39-FF4B4B.svg)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10-green.svg)](https://opencv.org/)

---

## 📖 Hakkında

Bu proje, el çizimi wireframe'leri modern, responsive web sitelerine dönüştürmek için Computer Vision ve Generative AI teknolojilerini birleştirir. LinkedIn portföy projesi olarak Senior Full-Stack AI Mühendisliği yeteneklerini sergiler.

---

## 🚀 Özellikler

### 🎯 Temel Özellikler
- ✅ **Computer Vision**: OpenCV ile görüntü ön işleme (Grayscale + Adaptive Thresholding)
- ✅ **Generative AI**: Google Gemini 2.5 Flash ile akıllı kod üretimi
- ✅ **Modern UI**: Streamlit ile kullanıcı dostu 4 sekme arayüz
- ✅ **Canlı Önizleme**: Oluşturulan web sitesini anında görüntüleme

### 🆕 Gelişmiş Özellikler

#### 1. 📊 Çoklu Görsel Desteği
- Birden fazla sayfa çizimi yükleme
- Her sayfa için ayrı kod üretimi
- Çok sayfalı web siteleri oluşturma

#### 2. 🔄 Versiyon Karşılaştırma
- Tek tıkla 3 farklı stil versiyonu oluşturma
- Yan yana karşılaştırma
- İstediğinizi seçip indirme

#### 3. 🛠️ Framework Seçimi
- **Tailwind CSS**: Utility-first modern yaklaşım
- **Bootstrap 5**: Popüler component library
- **Pure CSS**: Vanilla CSS, framework'sız

#### 4. 📱 Cihaz Önizleme
- 🖥️ Desktop görünümü (1920x1080)
- 📱 Tablet görünümü (768x1024)
- 📱 Mobile görünümü (375x667)
- Responsive test arayüzü

#### 5. 🎨 Renk Paleti Çıkarıcı
- KMeans clustering ile otomatik renk analizi
- Görselden 5 baskın renk çıkarma
- Çıkarılan renkleri tasarımda kullanma

#### 6. 💡 AI Öneri Sistemi
- Çizime bakarak akıllı öneriler
- Eksik özellikler hakkında bilgilendirme
- UX iyileştirme tavsiyeleri

#### 7. 🚀 SEO & Accessibility
- **SEO**: Meta tags, Open Graph, Twitter Cards, Semantic HTML5
- **Accessibility**: ARIA labels, alt texts, keyboard navigation, focus indicators
- Tek tıkla ekleme

#### 8. 📦 Gelişmiş Export
- **HTML**: Tek dosya olarak
- **ZIP**: Tüm dosyalar paketlenmiş (HTML + README)
- **React Component**: JSX formatında
- **Vue Component**: (Planlanan)

#### 9. 🔗 Sosyal Paylaşım
- QR kod oluşturma
- Embed kodu üretme
- Paylaşım URL'si

#### 10. 📚 Geçmiş/History
- Son 20 tasarımı otomatik kaydetme
- Favori işaretleme
- Geri yükleme
- Mini önizlemeler

#### 11. 🎨 Kişiselleştirme
- 5 renk paleti seçeneği
- 5 tasarım stili
- Responsive toggle
- Animasyon toggle
- Özel istek text alanı

---

## 📦 Kurulum

### Gereksinimler
- Python 3.8+
- pip

### Adımlar

1. **Repository'yi klonlayın veya indirin**

2. **Bağımlılıkları yükleyin:**

```bash
# Basit versiyon için
pip install -r requirements.txt

# Gelişmiş versiyon için
pip install -r requirements_advanced.txt
```

3. **Google API Key alın:**
   - [Google AI Studio](https://makersuite.google.com/app/apikey) adresine gidin
   - Ücretsiz API anahtarı oluşturun

---

## ▶️ Çalıştırma

### Basit Versiyon
```bash
streamlit run app.py
```

### Gelişmiş Versiyon (ÖNERİLEN)
```bash
streamlit run app_advanced.py
```

Tarayıcınızda otomatik olarak açılacaktır: `http://localhost:8501`

---

## 🎯 Kullanım Kılavuzu

### Adım 1: Kurulum
1. Uygulamayı başlatın
2. Sol sidebar'dan Google API Key'inizi girin

### Adım 2: Tasarım Tercihleri
3. Framework seçin (Tailwind/Bootstrap/Pure CSS)
4. Renk paleti ve stil seçin
5. Responsive/Animasyon tercihlerinizi belirleyin
6. İsteğe bağlı: SEO ve Accessibility ekleyin

### Adım 3: Görsel Yükleme
7. "Yeni Tasarım" sekmesinden çiziminizi yükleyin
8. Renk paletini inceleyin
9. (Opsiyonel) AI önerilerini alın

### Adım 4: Kod Oluşturma

**Tek Versiyon için:**
- "Kodu Oluştur" butonuna tıklayın

**Çoklu Versiyon için:**
- "3 Versiyon Oluştur" butonuna tıklayın
- "Versiyon Karşılaştır" sekmesine geçin

### Adım 5: Önizleme & Export
10. Önizleme sekmesinde canlı görüntüleyin
11. Kod sekmesinde inceleyip kopyalayın
12. Export sekmesinden istediğiniz formatta indirin
13. Paylaş sekmesinden QR kod/embed kod alın

### Adım 6: Cihaz Testleri
14. "Cihaz Önizleme" sekmesine geçin
15. Desktop, Tablet, Mobile görünümleri test edin

### Adım 7: Geçmiş
16. "Geçmiş" sekmesinden önceki tasarımlarınıza ulaşın
17. Favori işaretleyin veya geri yükleyin

---

## 🛠️ Teknolojiler

### Backend & AI
- **Python 3.11**: Ana programlama dili
- **Google Generative AI (Gemini 2.5 Flash)**: Kod üretimi
- **OpenCV**: Görüntü işleme
- **NumPy**: Matris operasyonları
- **Scikit-learn**: KMeans clustering (renk analizi)

### Frontend & UI
- **Streamlit**: Web arayüzü framework
- **Pillow (PIL)**: Görüntü formatları

### Utilities
- **qrcode**: QR kod oluşturma
- **zipfile**: Dosya paketleme
- **base64**: Encoding işlemleri
- **json**: Veri saklama

### Generated Code
- **Tailwind CSS / Bootstrap / Pure CSS**: Kullanıcı seçimine göre
- **HTML5 Semantic**: Modern, erişilebilir markup
- **Responsive Design**: Mobile-first yaklaşım

---

## 📸 Ekran Görüntüleri

### Ana Ekran
- Çoklu dosya yükleme
- Renk paleti çıkarma
- AI önerileri

### Versiyon Karşılaştırma
- 3 farklı stil yan yana
- Hızlı önizleme
- Tek tıkla indirme

### Cihaz Önizleme
- Desktop/Tablet/Mobile
- Gerçek boyutlarda test

### Geçmiş
- Timeline görünümü
- Favori sistemi
- Geri yükleme

---

## 💡 En İyi Sonuçlar İçin İpuçları

### Çizim Kalitesi
- ✏️ Net ve okunaklı çizin
- 📐 Ana bölümleri (header, content, footer) belirgin yapın
- 🔲 Kutuları ve bileşenleri işaretleyin
- 💡 İyi aydınlatmalı ortamda fotoğraf çekin
- 📄 Beyaz kağıt + siyah kalem ideal kombinasyon

### Element Örnekleri
- **Header**: Logo, navigasyon menüsü
- **Hero Section**: Büyük görsel, başlık, CTA butonu
- **Content**: Kartlar, grid layout, listeler
- **Footer**: İletişim, sosyal medya, copyright

### Özel İstekler
- "Animasyonlu hero section ekle"
- "3 sütunlu özellik kartları"
- "Sosyal medya iconları footer'da"
- "İletişim formu ile newsletter kaydı"
- "Sticky header navigation"

---

## 🔧 Özelleştirme

### API Model Değiştirme
```python
# app_advanced.py içinde
model = genai.GenerativeModel('gemini-2.5-flash')  # Mevcut

# Alternatifler:
# model = genai.GenerativeModel('gemini-2.5-pro')  # Daha güçlü
# model = genai.GenerativeModel('gemini-2.0-flash')  # Daha hızlı
```

### History Limiti
```python
# save_to_history fonksiyonunda
if len(st.session_state.history) > 20:  # 20'den fazla tutma
    # İstediğiniz sayıyı değiştirebilirsiniz
```

### Renk Sayısı
```python
# extract_color_palette fonksiyonunda
extracted_colors = extract_color_palette(image, n_colors=5)
# n_colors parametresini değiştirin
```

---

## 📊 Proje Yapısı

```
Sketch-to-Code/
├── app.py                      # Basit versiyon
├── app_advanced.py             # Gelişmiş versiyon (ÖNERİLEN)
├── test_api.py                 # API test scripti
├── requirements.txt            # Basit versiyon bağımlılıkları
├── requirements_advanced.txt   # Gelişmiş versiyon bağımlılıkları
└── README.md                   # Bu dosya
```

---

## 🐛 Sorun Giderme

### "API Key geçersiz" hatası
- API Key'in doğru kopyalandığından emin olun
- [Google AI Studio](https://makersuite.google.com/app/apikey)'da yeni key oluşturun
- API Key'in aktif olmasını bekleyin (5-10 dakika)

### "Model bulunamadı" hatası
- `test_api.py` scriptini çalıştırın
- Mevcut modelleri listeleyin
- `app_advanced.py` içinde model adını güncelleyin

### Görsel yüklenmiyor
- Dosya boyutunu kontrol edin (max 5MB önerilir)
- Desteklenen formatlar: JPG, PNG
- Görseli yeniden kaydetmeyi deneyin

### Streamlit çalışmıyor
```bash
pip uninstall streamlit
pip install streamlit==1.39.0
```

### Paket hatası
```bash
pip install --upgrade -r requirements_advanced.txt
```

---

## 🚀 Gelecek Özellikler (Roadmap)

- [ ] Vue.js component export
- [ ] Gerçek URL'de hosting ve canlı paylaşım
- [ ] Dark/Light mode toggle
- [ ] Çoklu dil desteği (İngilizce, Türkçe)
- [ ] Video tutorial entegrasyonu
- [ ] Template library
- [ ] Collaborative editing
- [ ] Version control (Git entegrasyonu)
- [ ] A/B testing için analitik
- [ ] AI ile otomatik bug fix

---

## 👨‍💻 Geliştirici

**LinkedIn Portföy Projesi**  
Senior Full-Stack AI Mühendisi

### Yetenekler Sergisi:
- ✅ Computer Vision (OpenCV)
- ✅ Generative AI (Google Gemini)
- ✅ Python Backend Development
- ✅ UI/UX Design (Streamlit)
- ✅ Machine Learning (KMeans)
- ✅ API Integration
- ✅ Data Visualization
- ✅ Clean Code & Documentation

---

## 📄 Lisans

Bu proje MIT lisansı altındadır. Eğitim ve portfolyo amaçlı kullanım için özgürdür.

---

## 🙏 Teşekkürler

- Google AI Studio - Ücretsiz Gemini API
- Streamlit - Harika web framework
- OpenCV - Güçlü görüntü işleme
- Tüm open-source katkıda bulunanlara

---

## 📞 İletişim

Proje hakkında sorularınız için:
- LinkedIn profilimden ulaşabilirsiniz
- Issue açabilirsiniz
- Pull request gönderebilirsiniz

---

**⭐ Projeyi beğendiyseniz yıldız vermeyi unutmayın!**

