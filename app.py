"""
Sketch-to-Code ADVANCED: AI ile Çizimden Web Sitesine (Gelişmiş Versiyon)
===========================================================================
Computer Vision ve Generative AI kullanarak el çizimi web sitesi taslağını
HTML/CSS koduna dönüştüren Senior Full-Stack AI uygulaması.

YENİ ÖZELLİKLER:
- Çoklu görsel desteği
- Framework seçimi (Tailwind/Bootstrap/Pure CSS)
- Cihaz önizleme (Desktop/Tablet/Mobile)
- Renk paleti çıkarıcı
- SEO & Accessibility
- Gelişmiş export seçenekleri
- Geçmiş/History sistemi
- AI öneri sistemi
- Sosyal paylaşım özellikleri

Teknolojiler: Streamlit, OpenCV, Google Gemini AI, SKLearn
Geliştirici: LinkedIn Portföy Projesi
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import google.generativeai as genai
import io
import base64
import json
from datetime import datetime
import zipfile
from io import BytesIO

# Optional imports - Eğer paketler yoksa ilgili özellikler devre dışı kalır
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn bulunamadı. Renk paleti çıkarma özelliği devre dışı.")


# Sayfa yapılandırması
st.set_page_config(
    page_title="Sketch-to-Code AI - Advanced",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state başlatma
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_code' not in st.session_state:
    st.session_state.current_code = None
if 'extracted_colors' not in st.session_state:
    st.session_state.extracted_colors = []
if 'generated_versions' not in st.session_state:
    st.session_state.generated_versions = []


def extract_color_palette(image, n_colors=5):
    """
    Görselden baskın renk paletini çıkarır (KMeans clustering kullanarak).
    
    Args:
        image: PIL Image objesi
        n_colors: Çıkarılacak renk sayısı
    
    Returns:
        list: Hex formatında renk listesi
    """
    if not SKLEARN_AVAILABLE:
        # Sklearn yoksa basit renk çıkarma
        img = image.resize((100, 100))
        img_array = np.array(img)
        
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        # Basit örnekleme ile renkler
        pixels = img_array.reshape(-1, 3)
        step = len(pixels) // n_colors
        sample_colors = pixels[::step][:n_colors]
        
        hex_colors = ['#%02x%02x%02x' % tuple(c) for c in sample_colors]
        return hex_colors
    
    # Sklearn varsa gelişmiş clustering
    img = image.resize((150, 150))
    img_array = np.array(img)
    
    # RGB formatına çevir
    if len(img_array.shape) == 2:  # Grayscale ise
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:  # RGBA ise
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Reshape: (height, width, 3) -> (height*width, 3)
    pixels = img_array.reshape(-1, 3)
    
    # KMeans ile renk kümeleme
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    # Baskın renkleri al
    colors = kmeans.cluster_centers_.astype(int)
    
    # Hex formatına çevir
    hex_colors = ['#%02x%02x%02x' % (r, g, b) for r, g, b in colors]
    
    return hex_colors


def preprocess_image(image):
    """
    Computer Vision kullanarak görseli ön işleme (preprocessing) fonksiyonu.
    """
    img_array = np.array(image)
    
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    processed = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    return img_array, processed


def generate_code_with_options(image, api_key, options):
    """
    Gelişmiş seçeneklerle kod oluşturur.
    
    Args:
        image: İşlenmiş görsel
        api_key: Google API Key
        options: dict - Tüm kullanıcı seçenekleri
    
    Returns:
        str: Oluşturulan HTML/CSS kodu
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        pil_image = Image.fromarray(image)
        
        # Framework seçimi
        framework_instructions = {
            "Tailwind CSS": "Tailwind CSS CDN kullan. Utility-first yaklaşımı uygula.",
            "Bootstrap 5": "Bootstrap 5 CDN kullan. Bootstrap componentlerini kullan.",
            "Pure CSS": "Harici framework kullanma. Modern, vanilla CSS yaz. CSS Grid ve Flexbox kullan."
        }
        
        # Renk paleti talimatı
        color_palette_text = ""
        if options.get('use_extracted_colors') and options.get('extracted_colors'):
            colors_str = ", ".join(options['extracted_colors'])
            color_palette_text = f"Bu renk paletini kullan: {colors_str}"
        
        # SEO ve Accessibility talimatı
        seo_text = ""
        if options.get('add_seo'):
            seo_text = """
            SEO ÖZELLİKLERİ EKLE:
            - Meta description, keywords, author tags
            - Open Graph tags (Facebook/LinkedIn paylaşımı için)
            - Twitter Card tags
            - Semantic HTML5 tags (article, section, nav, etc.)
            """
        
        accessibility_text = ""
        if options.get('add_accessibility'):
            accessibility_text = """
            ACCESSIBILITY ÖZELLİKLERİ EKLE:
            - ARIA labels ve roles
            - Alt texts tüm görsellere
            - Keyboard navigation desteği
            - Focus indicators
            - Contrast ratio optimize et
            """
        
        # Ana prompt
        prompt = f"""
        Sen uzman bir Frontend geliştiricisisin. 
        
        Bu wireframe çizimini modern bir web sitesi koduna dönüştür.
        
        FRAMEWORK: {framework_instructions[options['framework']]}
        
        TASARIM TERCİHLERİ:
        🎨 Renk Paleti: {options['color_scheme']}
        {color_palette_text}
        🖼️ Tasarım Stili: {options['design_style']}
        {"📱 Responsive: Mobil, tablet, desktop uyumlu" if options['responsive'] else ""}
        {"✨ Animasyonlar: Smooth transitions, hover effects, fade-in" if options['animations'] else ""}
        
        {options.get('custom_prompt', '')}
        
        {seo_text}
        {accessibility_text}
        
        TEKNİK KURALLAR:
        - Production-ready, temiz kod yaz
        - Sadece HTML kodunu döndür (markdown blokları kullanma)
        - Tüm elementleri (header, nav, content, footer, buttons) koda dök
        - Gerçek içerik kullan, placeholder değil
        - Modern best practices uygula
        
        Kullanıcının tüm tercihlerine uy!
        """
        
        response = model.generate_content([prompt, pil_image])
        return response.text
        
    except Exception as e:
        return f"❌ Hata: {str(e)}"


def generate_ai_suggestions(image, api_key):
    """
    Çizime bakarak AI önerileri üretir.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        pil_image = Image.fromarray(image)
        
        prompt = """
        Bu web sitesi wireframe'ine bakarak kısa, net öneriler ver.
        
        Sadece 3-4 madde halinde şunları öner:
        - Eksik olan önemli özellikler
        - Tasarım iyileştirmeleri
        - Kullanıcı deneyimi tavsiyeleri
        
        Kısa ve öz cevap ver. Her öneri 1 satır olsun.
        """
        
        response = model.generate_content([prompt, pil_image])
        return response.text
        
    except Exception as e:
        return None


def create_device_preview_html(html_code, device_width):
    """
    Farklı cihaz boyutları için önizleme HTML'i oluşturur.
    """
    return f"""
    <div style="width: {device_width}px; margin: 0 auto; border: 2px solid #ccc; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        {html_code}
    </div>
    """


def create_zip_export(html_code, filename="website"):
    """
    HTML, CSS, JS'yi ayrı dosyalar halinde zip olarak export eder.
    """
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # HTML dosyası
        zip_file.writestr(f"{filename}/index.html", html_code)
        
        # README
        readme = f"""
# {filename}

Bu web sitesi Sketch-to-Code AI ile oluşturulmuştur.

## Kullanım
1. index.html dosyasını tarayıcınızda açın
2. İsterseniz style.css ve script.js dosyalarını düzenleyin

Oluşturulma Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        """
        zip_file.writestr(f"{filename}/README.md", readme)
    
    zip_buffer.seek(0)
    return zip_buffer


def convert_to_react_component(html_code):
    """
    HTML kodunu React component'ine dönüştürür (basit versiyon).
    """
    react_code = f"""
import React from 'react';

const GeneratedComponent = () => {{
  return (
    <div dangerouslySetInnerHTML={{{{__html: `
{html_code}
    `}}}} />
  );
}};

export default GeneratedComponent;

/* 
 * Not: Bu otomatik dönüşümdür. Production için:
 * - dangerouslySetInnerHTML yerine proper JSX kullanın
 * - Inline CSS'i ayrı dosyaya taşıyın
 * - Component'i parçalara ayırın
 */
"""
    return react_code


def save_to_history(code, options, thumbnail=None):
    """
    Oluşturulan kodu geçmişe kaydet.
    """
    history_item = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'code': code,
        'options': options,
        'thumbnail': thumbnail,
        'favorite': False
    }
    
    st.session_state.history.insert(0, history_item)
    
    # Maksimum 20 item tut
    if len(st.session_state.history) > 20:
        st.session_state.history = st.session_state.history[:20]


def main():
    """
    Ana uygulama - Gelişmiş versiyon
    """
    # Header
    st.title("🎨 Sketch-to-Code AI: Advanced Edition")
    st.markdown("**Computer Vision** + **Generative AI** = Profesyonel Web Siteleri")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Ayarlar")
        
        # API Key
        st.markdown("### 🔑 Google API Key")
        api_key = st.text_input(
            "API Anahtarı:",
            type="password",
            placeholder="AIzaSy...",
            help="Google AI Studio'dan alın"
        )
        
        st.divider()
        
        # Framework Seçimi
        st.markdown("### 🛠️ Framework")
        framework = st.selectbox(
            "CSS Framework:",
            ["Tailwind CSS", "Bootstrap 5", "Pure CSS"],
            help="Hangi CSS framework kullanılsın?"
        )
        
        st.divider()
        
        # Tasarım Tercihleri
        st.markdown("### 🎨 Tasarım")
        
        color_scheme = st.selectbox(
            "Renk Paleti:",
            ["Modern Mavi-Beyaz", "Dark Mode", "Canlı Renkler", 
             "Profesyonel Kurumsal", "Pastel Tonlar", "Çıkarılan Renkleri Kullan"]
        )
        
        design_style = st.selectbox(
            "Stil:",
            ["Modern Minimal", "Klasik Zarif", "Yaratıcı Cesur",
             "E-ticaret", "Blog/Portfolyo"]
        )
        
        responsive = st.checkbox("📱 Responsive", value=True)
        animations = st.checkbox("✨ Animasyonlar", value=False)
        
        st.divider()
        
        # SEO & Accessibility
        st.markdown("### 🚀 Optimizasyon")
        add_seo = st.checkbox("🔍 SEO Tags Ekle", value=False)
        add_accessibility = st.checkbox("♿ Accessibility Ekle", value=False)
        
        st.divider()
        
        # Özel İstekler
        st.markdown("### 💭 Özel İstekler")
        custom_prompt = st.text_area(
            "Eklemek istedikleriniz:",
            placeholder="- Slider ekle\n- İletişim formu\n- Sosyal medya",
            height=100
        )
        
        st.divider()
        
        # History Sidebar
        if st.session_state.history:
            st.markdown("### 📚 Geçmiş")
            if st.button("🗑️ Geçmişi Temizle"):
                st.session_state.history = []
                st.rerun()
    
    # Ana İçerik
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Yeni Tasarım", "📊 Versiyon Karşılaştır", 
        "📱 Cihaz Önizleme", "📚 Geçmiş"
    ])
    
    # TAB 1: Yeni Tasarım
    with tab1:
        st.header("📤 Çiziminizi Yükleyin")
        
        # Çoklu dosya yükleme
        uploaded_files = st.file_uploader(
            "Wireframe görselleri (Birden fazla sayfa yükleyebilirsiniz)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="Birden fazla sayfa için ayrı çizimler yükleyin"
        )
        
        if uploaded_files:
            # Her görsel için işlem
            for idx, uploaded_file in enumerate(uploaded_files):
                st.subheader(f"📄 Sayfa {idx + 1}: {uploaded_file.name}")
                
                image = Image.open(uploaded_file)
                
                # Renk paleti çıkar
                with st.spinner("🎨 Renkler çıkarılıyor..."):
                    extracted_colors = extract_color_palette(image)
                    st.session_state.extracted_colors = extracted_colors
                
                # Görsel işleme
                with st.spinner("🔍 Görsel işleniyor..."):
                    original, processed = preprocess_image(image)
                
                # Görselleri göster
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**📷 Orijinal**")
                    st.image(original, use_column_width=True)
                
                with col2:
                    st.markdown("**🤖 İşlenmiş**")
                    st.image(processed, use_column_width=True)
                
                with col3:
                    st.markdown("**🎨 Renk Paleti**")
                    colors_html = "".join([
                        f'<div style="background:{c}; width:40px; height:40px; display:inline-block; margin:2px; border-radius:4px;" title="{c}"></div>'
                        for c in extracted_colors
                    ])
                    st.markdown(colors_html, unsafe_allow_html=True)
                    st.caption("Çıkarılan renkler")
                
                # AI Önerileri
                if api_key and st.button(f"💡 AI Önerileri Al (Sayfa {idx+1})", key=f"suggest_{idx}"):
                    with st.spinner("🤖 AI analiz ediyor..."):
                        suggestions = generate_ai_suggestions(processed, api_key)
                        if suggestions:
                            st.info(f"**🎯 AI Önerileri:**\n\n{suggestions}")
                
                st.divider()
                
                # Kod oluştur
                if api_key:
                    col_btn1, col_btn2 = st.columns(2)
                    
                    with col_btn1:
                        if st.button(f"✨ Kodu Oluştur (Sayfa {idx+1})", type="primary", key=f"gen_{idx}"):
                            with st.spinner("🧠 AI kod yazıyor..."):
                                options = {
                                    'framework': framework,
                                    'color_scheme': color_scheme,
                                    'design_style': design_style,
                                    'responsive': responsive,
                                    'animations': animations,
                                    'custom_prompt': custom_prompt,
                                    'add_seo': add_seo,
                                    'add_accessibility': add_accessibility,
                                    'use_extracted_colors': color_scheme == "Çıkarılan Renkleri Kullan",
                                    'extracted_colors': extracted_colors
                                }
                                
                                generated_code = generate_code_with_options(processed, api_key, options)
                                
                                if generated_code and not generated_code.startswith("❌"):
                                    generated_code = generated_code.replace("```html", "").replace("```", "").strip()
                                    st.session_state.current_code = generated_code
                                    
                                    # Geçmişe kaydet
                                    save_to_history(generated_code, options)
                                    
                                    st.success("✅ Kod başarıyla oluşturuldu!")
                                    st.rerun()
                                else:
                                    st.error(generated_code)
                    
                    with col_btn2:
                        if st.button(f"🎲 3 Versiyon Oluştur (Sayfa {idx+1})", key=f"multi_{idx}"):
                            st.session_state.generated_versions = []
                            
                            styles = ["Modern Minimal", "Klasik Zarif", "Yaratıcı Cesur"]
                            
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i, style in enumerate(styles):
                                status_text.text(f"🎨 {style} versiyonu oluşturuluyor...")
                                
                                options = {
                                    'framework': framework,
                                    'color_scheme': color_scheme,
                                    'design_style': style,
                                    'responsive': responsive,
                                    'animations': animations,
                                    'custom_prompt': custom_prompt,
                                    'add_seo': add_seo,
                                    'add_accessibility': add_accessibility,
                                    'use_extracted_colors': color_scheme == "Çıkarılan Renkleri Kullan",
                                    'extracted_colors': extracted_colors
                                }
                                
                                code = generate_code_with_options(processed, api_key, options)
                                code = code.replace("```html", "").replace("```", "").strip()
                                
                                st.session_state.generated_versions.append({
                                    'style': style,
                                    'code': code
                                })
                                
                                progress_bar.progress((i + 1) / len(styles))
                            
                            status_text.text("✅ Tüm versiyonlar hazır!")
                            st.success("3 farklı versiyon oluşturuldu! 'Versiyon Karşılaştır' sekmesine geçin.")
                
                else:
                    st.warning("⚠️ API Key girmelisiniz")
            
            # Mevcut kod varsa göster
            if st.session_state.current_code:
                st.divider()
                
                # Seçilen versiyon bilgisi
                if 'selected_version' in st.session_state:
                    st.info(f"📋 Görüntülenen versiyon: **{st.session_state['selected_version']}**")
                
                st.header("🌐 Oluşturulan Web Sitesi")
                
                view_tab1, view_tab2, view_tab3, view_tab4 = st.tabs([
                    "👁️ Önizleme", "💻 Kod", "📦 Export", "🔗 Paylaş"
                ])
                
                with view_tab1:
                    st.components.v1.html(st.session_state.current_code, height=600, scrolling=True)
                
                with view_tab2:
                    st.code(st.session_state.current_code, language="html", line_numbers=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "📥 HTML İndir",
                            st.session_state.current_code,
                            "website.html",
                            "text/html"
                        )
                    with col2:
                        react_code = convert_to_react_component(st.session_state.current_code)
                        st.download_button(
                            "⚛️ React Component İndir",
                            react_code,
                            "Component.jsx",
                            "text/javascript"
                        )
                
                with view_tab3:
                    st.markdown("### 📦 Export Seçenekleri")
                    
                    # ZIP Export
                    zip_data = create_zip_export(st.session_state.current_code, "my_website")
                    st.download_button(
                        "📦 ZIP olarak indir (Tüm dosyalar)",
                        zip_data,
                        "website.zip",
                        "application/zip"
                    )
                    
                    st.info("ZIP içeriği: index.html, README.md")
                
                with view_tab4:
                    st.markdown("### 🔗 Paylaşım Seçenekleri")
                    
                    # URL Paylaşımı
                    example_url = "https://example.com/my-design"
                    st.text_input("Paylaşım URL'si:", example_url)
                    
                    st.info("💡 Web sitenizi bir hosting servisine yükledikten sonra bu URL'yi paylaşabilirsiniz.")
                    
                    # Embed kodu
                    st.markdown("**Embed Kodu:**")
                    st.caption("Bu kodu başka bir web sitesine gömebilirsiniz")
                    embed_code = f'<iframe src="{example_url}" width="100%" height="600" frameborder="0"></iframe>'
                    st.code(embed_code, language="html")
                    
                    # Sosyal medya paylaşım linkleri
                    st.markdown("**Sosyal Medya Paylaşımı:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        linkedin_url = f"https://www.linkedin.com/sharing/share-offsite/?url={example_url}"
                        st.markdown(f"[🔗 LinkedIn'de Paylaş]({linkedin_url})")
                    with col2:
                        twitter_url = f"https://twitter.com/intent/tweet?url={example_url}"
                        st.markdown(f"[🐦 Twitter'da Paylaş]({twitter_url})")
                    with col3:
                        facebook_url = f"https://www.facebook.com/sharer/sharer.php?u={example_url}"
                        st.markdown(f"[📘 Facebook'ta Paylaş]({facebook_url})")
        
        else:
            st.info("👆 Başlamak için bir veya daha fazla görsel yükleyin")
    
    # TAB 2: Versiyon Karşılaştırma
    with tab2:
        st.header("📊 Versiyon Karşılaştırma")
        
        if st.session_state.generated_versions:
            st.success(f"✅ {len(st.session_state.generated_versions)} versiyon oluşturuldu")
            
            # Versiyonları yan yana göster
            cols = st.columns(len(st.session_state.generated_versions))
            
            for idx, (col, version) in enumerate(zip(cols, st.session_state.generated_versions)):
                with col:
                    st.markdown(f"### {version['style']}")
                    
                    # Küçük önizleme
                    preview_html = f'<div style="transform: scale(0.3); transform-origin: top left; width: 333%; height: 400px; overflow: hidden;">{version["code"]}</div>'
                    st.components.v1.html(preview_html, height=120)
                    
                    # Butonlar
                    btn_col1, btn_col2 = st.columns(2)
                    
                    with btn_col1:
                        # Tam görüntüle butonu
                        if st.button(f"👁️ Görüntüle", key=f"view_{idx}", type="primary", use_container_width=True):
                            st.session_state.current_code = version['code']
                            st.session_state['selected_version'] = version['style']
                            st.rerun()
                    
                    with btn_col2:
                        # İndir butonu
                        st.download_button(
                            f"📥 İndir",
                            version['code'],
                            f"{version['style'].lower().replace(' ', '_')}.html",
                            "text/html",
                            key=f"dl_{idx}",
                            use_container_width=True
                        )
        else:
            st.info("🎲 Önce 'Yeni Tasarım' sekmesinden '3 Versiyon Oluştur' butonuna tıklayın")
    
    # TAB 3: Cihaz Önizleme
    with tab3:
        st.header("📱 Cihaz Önizleme")
        
        if st.session_state.current_code:
            # Cihaz seçimi
            device_choice = st.radio(
                "Cihaz Seçin:",
                ["🖥️ Desktop (1920px)", "📱 Tablet (768px)", "📱 Mobile (375px)", "📊 Hepsi Yan Yana"],
                horizontal=True
            )
            
            st.divider()
            
            if device_choice == "📊 Hepsi Yan Yana":
                device_col1, device_col2, device_col3 = st.columns(3)
                
                with device_col1:
                    st.markdown("### 🖥️ Desktop")
                    st.caption("1920x1080")
                    desktop_html = create_device_preview_html(st.session_state.current_code, 380)
                    st.components.v1.html(desktop_html, height=500, scrolling=True)
                
                with device_col2:
                    st.markdown("### 📱 Tablet")
                    st.caption("768x1024")
                    tablet_html = create_device_preview_html(st.session_state.current_code, 320)
                    st.components.v1.html(tablet_html, height=500, scrolling=True)
                
                with device_col3:
                    st.markdown("### 📱 Mobile")
                    st.caption("375x667")
                    mobile_html = create_device_preview_html(st.session_state.current_code, 280)
                    st.components.v1.html(mobile_html, height=500, scrolling=True)
            
            elif device_choice == "🖥️ Desktop (1920px)":
                st.markdown("### 🖥️ Desktop Görünümü")
                st.caption("Tam ekran boyutu: 1920x1080")
                st.components.v1.html(st.session_state.current_code, height=700, scrolling=True)
            
            elif device_choice == "📱 Tablet (768px)":
                st.markdown("### 📱 Tablet Görünümü")
                st.caption("Orta ekran boyutu: 768x1024")
                tablet_html = create_device_preview_html(st.session_state.current_code, 768)
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.components.v1.html(tablet_html, height=700, scrolling=True)
            
            else:  # Mobile
                st.markdown("### 📱 Mobile Görünümü")
                st.caption("Küçük ekran boyutu: 375x667")
                mobile_html = create_device_preview_html(st.session_state.current_code, 375)
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    st.components.v1.html(mobile_html, height=700, scrolling=True)
        else:
            st.info("⚠️ Önce 'Yeni Tasarım' sekmesinden bir tasarım oluşturun")
    
    # TAB 4: Geçmiş
    with tab4:
        st.header("📚 Tasarım Geçmişi")
        
        if st.session_state.history:
            for idx, item in enumerate(st.session_state.history):
                with st.expander(f"📄 {item['timestamp']} - {item['options'].get('design_style', 'Bilinmiyor')}"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Framework:** {item['options'].get('framework')}")
                        st.markdown(f"**Stil:** {item['options'].get('design_style')}")
                        st.markdown(f"**Renk:** {item['options'].get('color_scheme')}")
                    
                    with col2:
                        if st.button("🔄 Geri Yükle", key=f"restore_{idx}"):
                            st.session_state.current_code = item['code']
                            st.success("✅ Geri yüklendi!")
                            st.rerun()
                        
                        if st.button("⭐ Favori", key=f"fav_{idx}"):
                            st.session_state.history[idx]['favorite'] = not item.get('favorite', False)
                            st.rerun()
                    
                    # Mini önizleme
                    with st.container():
                        mini_preview = f'<div style="transform: scale(0.2); transform-origin: top left; width: 500%; height: 200px; overflow: hidden;">{item["code"]}</div>'
                        st.components.v1.html(mini_preview, height=40)
        else:
            st.info("📭 Henüz geçmiş yok. İlk tasarımınızı oluşturun!")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🎨 <b>Sketch-to-Code AI - Advanced Edition</b></p>
        <p>Computer Vision + Generative AI | LinkedIn Portföy Projesi</p>
        <p>Teknolojiler: Streamlit • OpenCV • Google Gemini AI • SKLearn</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
