# Student Spending Analysis

## Proje Hakkında
Bu projede üniversite öğrencilerinin gelir yapıları ve harcama alışkanlıkları incelenmiştir. Amaç, öğrencilerin harcama davranışlarının gerçekten hangi faktörlere bağlı olduğunu anlamaktır.

Analiz sürecinde veri keşfi (EDA), özellik mühendisliği, segmentasyon, makine öğrenmesi ve model yorumlama adımları uygulanmıştır.

---

## Veri Seti
Çalışmada 1000 gözlemden oluşan bir veri seti kullanılmıştır.

Veri setinde şu bilgiler yer almaktadır:

- Demografik bilgiler: yaş, cinsiyet, bölüm, sınıf  
- Finansal bilgiler: aylık gelir, finansal destek, okul ücreti  
- Harcama kalemleri: barınma, gıda, ulaşım, kitap, eğlence, kişisel bakım, teknoloji, sağlık  
- Ödeme yöntemi  

Veri setinde eksik değer bulunmamaktadır.

---

## Yapılan Analizler

### 1. Keşifsel Veri Analizi (EDA)
- Gelir ve harcama dağılımları incelendi  
- Cinsiyet ve sınıf bazlı karşılaştırmalar yapıldı  
- Korelasyon matrisi oluşturuldu  
- IQR yöntemi ile aykırı değer kontrolü yapıldı  

---

### 2. Özellik Mühendisliği
- **Efektif gelir** oluşturuldu (gelir + finansal destek)  
- Tüm harcama kalemleri toplanarak **toplam harcama** değişkeni üretildi  
- Harcama oranları hesaplanarak öğrencilerin bütçe dağılımı analiz edildi  

---

### 3. Segmentasyon
Öğrenciler gelir ve harcama seviyelerine göre gruplandırıldı.

Elde edilen en önemli sonuç:

> Öğrencilerin harcama seviyeleri gelirden bağımsız olarak benzer kalmaktadır.

Bu durum, harcamanın gelirden çok ihtiyaçlara bağlı olduğunu göstermektedir.

---

### 4. Makine Öğrenmesi
İki farklı model denendi:

- Linear Regression → düşük performans (R² ≈ 0.04)  
- Random Forest → daha iyi performans (R² ≈ 0.46)  

Bu sonuç, değişkenler arasındaki ilişkinin lineer olmadığını göstermektedir.

---

### 5. Feature Importance
Random Forest modeline göre:

- Harcamanın nasıl dağıldığı (özellikle barınma oranı) en önemli faktör  
- Gelir ve finansal destek daha düşük etkili  

---

### 6. SHAP Analizi
SHAP analizi ile modelin karar mekanizması incelendi.

Elde edilen sonuç:

> Harcama davranışı tek bir değişkenle açıklanamıyor.

Aynı gelir seviyesine sahip öğrenciler farklı harcama davranışları gösterebilmektedir.

---

## Temel Bulgular

- Öğrencilerin harcamalarının büyük kısmı zorunlu giderlerden oluşmaktadır  
- Harcama seviyesi gelirden bağımsızdır  
- Düşük gelirli öğrenciler finansal baskı altındadır  
- Ödeme yöntemi ve okul ücreti harcamayı anlamlı şekilde etkilememektedir  
- Harcama davranışı lineer değil, daha karmaşık bir yapıdadır  

---

## Genel Sonuç

Bu çalışma göstermektedir ki:

> Öğrenci harcama davranışı, gelirden çok zorunlu ihtiyaçlar ve yaşam maliyeti tarafından belirlenmektedir.

Bu nedenle öğrenci ekonomisi, esnek bir tüketim yapısından ziyade sabit giderler üzerine kurulu bir sistem olarak değerlendirilebilir.

---

## Kullanılan Teknolojiler

- Python  
- Pandas  
- Seaborn / Matplotlib  
- Scikit-learn  
- SHAP  

---

## Proje Yapısı
