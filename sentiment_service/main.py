"""
================================================================================
BrandGuard AI - Sentiment Analysis Service
================================================================================
Proje: Marka İtibar Yönetimi Mikroservis Mimarisi
Amaç: Müşteri yorumlarını NLP (Natural Language Processing) teknikleri ile
      analiz ederek duygu durumu tespiti yapan ve sonuçları DynamoDB'ye
      kaydeden mikroservis.

Mimari:
  - FastAPI tabanlı RESTful API
  - TextBlob kütüphanesi ile sentiment analysis
  - AWS DynamoDB entegrasyonu (Boto3)
  - Graceful degradation: AWS credentials yoksa mock save yapar

Yazar: [Öğrenci Adı]
Tarih: 2024
================================================================================
"""

from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from textblob import TextBlob
import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError, ClientError
import logging

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# PYDANTIC MODELLERİ
# ============================================================================


class Review(BaseModel):
    """
    Müşteri yorumu için giriş veri modeli.
    
    Attributes:
        brand (str): Analiz edilecek marka adı
        text (str): Müşteri yorumu metni
    """
    brand: str
    text: str

    @field_validator("brand", "text")
    @classmethod
    def not_empty(cls, v: str) -> str:
        """
        Alan validasyonu: Boş string kontrolü.
        
        Args:
            v: Validasyon yapılacak string değer
            
        Returns:
            str: Temizlenmiş string
            
        Raises:
            ValueError: Alan boş veya sadece boşluk içeriyorsa
        """
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


class SentimentResult(BaseModel):
    """
    Sentiment analizi sonucu için çıkış veri modeli.
    
    Attributes:
        brand (str): Analiz edilen marka adı
        text (str): Analiz edilen yorum metni
        sentiment (str): Duygu durumu etiketi ("POSITIVE", "NEUTRAL", "CRITICAL")
        score (float): Polarity skoru (-1.0 ile 1.0 arası)
        timestamp (str): Analiz zamanı (ISO 8601 formatında UTC)
    """
    brand: str
    text: str
    sentiment: str
    score: float
    timestamp: str


# ============================================================================
# FASTAPI UYGULAMA
# ============================================================================

app = FastAPI(
    title="BrandGuard AI - Sentiment Analysis Service",
    description="Müşteri yorumlarını NLP ile analiz eden ve DynamoDB'ye kaydeden servis",
    version="1.0.0"
)

# DynamoDB tablo adı (environment variable'dan alınabilir)
DYNAMODB_TABLE_NAME = "brandguard-reviews"


# ============================================================================
# SENTIMENT ANALYSIS FONKSİYONLARI
# ============================================================================


def analyze_sentiment(text: str) -> tuple[str, float]:
    """
    TextBlob kütüphanesi kullanarak metnin duygu durumunu analiz eder.
    
    TextBlob, metni tokenize edip her kelime için önceden eğitilmiş
    sentiment lexicon'larını kullanarak genel bir polarity skoru hesaplar.
    
    Polarity Skoru:
        - -1.0: Çok negatif (örn: "terrible", "awful")
        -  0.0: Nötr (örn: "okay", "fine")
        - +1.0: Çok pozitif (örn: "excellent", "amazing")
    
    Sentiment Kategorileri:
        - CRITICAL: polarity < -0.1 (Kritik şikayetler, acil müdahale gerekir)
        - NEUTRAL: -0.1 <= polarity <= 0.1 (Nötr yorumlar, risk yok)
        - POSITIVE: polarity > 0.1 (Olumlu yorumlar, marka değeri artışı)
    
    Args:
        text (str): Analiz edilecek metin
        
    Returns:
        tuple[str, float]: (sentiment_label, polarity_score)
            - sentiment_label: "CRITICAL", "NEUTRAL" veya "POSITIVE"
            - polarity_score: -1.0 ile 1.0 arası float değer
            
    Example:
        >>> analyze_sentiment("This product is amazing!")
        ('POSITIVE', 0.8)
        
        >>> analyze_sentiment("Terrible service, very disappointed")
        ('CRITICAL', -0.6)
    """
    blob = TextBlob(text)
    polarity = float(blob.sentiment.polarity)
    
    # Eşik değerlerine göre kategorilendirme
    if polarity < -0.1:
        label = "CRITICAL"
    elif polarity > 0.1:
        label = "POSITIVE"
    else:
        label = "NEUTRAL"
    
    logger.info(f"Sentiment analysis: {label} (score: {polarity:.3f})")
    return label, polarity


# ============================================================================
# DYNAMODB ENTEGRASYONU
# ============================================================================


def save_to_dynamodb(item: dict) -> None:
    """
    Analiz sonucunu AWS DynamoDB'ye kaydetmeye çalışır.
    
    Graceful Degradation Pattern:
        - AWS credentials varsa ve DynamoDB erişilebilirse: Gerçek kayıt yapar
        - AWS credentials yoksa veya hata oluşursa: Mock save yapar (print)
        - Bu sayede servis AWS olmadan da çalışabilir (development/testing)
    
    DynamoDB Tablo Yapısı:
        - Primary Key: brand (String)
        - Sort Key: timestamp (String)
        - Attributes: text, sentiment, score
    
    Args:
        item (dict): DynamoDB'ye kaydedilecek veri
            {
                "brand": str,
                "text": str,
                "sentiment": str,
                "score": float,
                "timestamp": str
            }
            
    Note:
        Production ortamında burada structured logging (CloudWatch, etc.)
        kullanılmalıdır. Şu an basit print kullanılıyor.
    """
    try:
        # Tablo adını item'dan çıkar (opsiyonel override için)
        table_name = item.pop("_table_name", DYNAMODB_TABLE_NAME)
        
        # Boto3 ile DynamoDB resource oluştur
        # Credentials environment variable'lardan otomatik alınır:
        # AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
        dynamodb = boto3.resource("dynamodb")
        table = dynamodb.Table(table_name)
        
        # Item'ı DynamoDB'ye yaz
        table.put_item(Item=item)
        
        logger.info(f"Successfully saved to DynamoDB: {item.get('brand')}")
        print(f"DynamoDB'ye yazıldı: {item.get('brand')}")
        
    except NoCredentialsError:
        # AWS credentials bulunamadı (development ortamı)
        logger.warning("AWS credentials not found. Using mock save.")
        print(f"DynamoDB Mock Save: {item.get('brand')} | Reason: No AWS credentials")
        
    except ClientError as e:
        # AWS API hatası (tablo yok, izin yok, vb.)
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        logger.error(f"DynamoDB ClientError ({error_code}): {e}")
        print(f"DynamoDB Mock Save: {item.get('brand')} | Reason: {error_code}")
        
    except BotoCoreError as e:
        # Boto3 core hatası (network, vb.)
        logger.error(f"DynamoDB BotoCoreError: {e}")
        print(f"DynamoDB Mock Save: {item.get('brand')} | Reason: {type(e).__name__}")
        
    except Exception as e:
        # Beklenmeyen hatalar
        logger.error(f"Unexpected error while saving to DynamoDB: {e}")
        print(f"DynamoDB Mock Save: {item.get('brand')} | Reason: {type(e).__name__}")


# ============================================================================
# API ENDPOINT'LERİ
# ============================================================================


@app.post("/analyze", response_model=SentimentResult)
async def analyze(review: Review) -> SentimentResult:
    """
    Müşteri yorumunu alır, sentiment analizi yapar, DynamoDB'ye kaydeder
    ve sonucu JSON formatında döner.
    
    İş Akışı:
        1. Gelen veri Pydantic modeli ile validasyondan geçer
        2. TextBlob ile sentiment analizi yapılır (polarity hesaplanır)
        3. Polarity skoruna göre kategori belirlenir (CRITICAL/NEUTRAL/POSITIVE)
        4. UTC timestamp oluşturulur
        5. Sonuç DynamoDB'ye kaydedilmeye çalışılır (başarısız olursa mock save)
        6. Sonuç JSON olarak döndürülür
    
    Args:
        review (Review): Analiz edilecek müşteri yorumu
        
    Returns:
        SentimentResult: Analiz sonucu
            {
                "brand": str,
                "text": str,
                "sentiment": "CRITICAL" | "NEUTRAL" | "POSITIVE",
                "score": float,
                "timestamp": str (ISO 8601 UTC)
            }
            
    Raises:
        HTTPException:
            - 422: Validasyon hatası (Pydantic otomatik döner)
    """
    logger.info(f"Analyzing review for brand: {review.brand}")
    
    # Sentiment analizi yap
    sentiment_label, score = analyze_sentiment(review.text)
    
    # UTC timestamp oluştur (ISO 8601 formatında)
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # Sonuç modelini oluştur
    result = SentimentResult(
        brand=review.brand,
        text=review.text,
        sentiment=sentiment_label,
        score=score,
        timestamp=timestamp,
    )
    
    # DynamoDB'ye kaydet (veya mock save)
    save_to_dynamodb(
        {
            "_table_name": DYNAMODB_TABLE_NAME,  # Opsiyonel override için
            "brand": result.brand,
            "text": result.text,
            "sentiment": result.sentiment,
            "score": result.score,
            "timestamp": result.timestamp,
        }
    )
    
    logger.info(f"Analysis completed for brand: {review.brand} - {sentiment_label}")
    return result


@app.get("/health")
async def health_check() -> dict:
    """
    Servis sağlık kontrolü endpoint'i.
    Docker/Kubernetes gibi orchestrator'lar için kullanılır.
    
    Returns:
        dict: Servis durumu bilgisi
    """
    return {"status": "healthy", "service": "sentiment_service"}


# ============================================================================
# UYGULAMA BAŞLATMA
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Development modunda çalıştırma
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
