"""
================================================================================
BrandGuard AI - Ingestion Service
================================================================================
Proje: Marka İtibar Yönetimi Mikroservis Mimarisi
Amaç: Müşteri yorumlarını toplayıp Sentiment Analysis servisine yönlendiren
      API Gateway katmanı görevi gören mikroservis.

Mimari:
  - FastAPI tabanlı RESTful API
  - Pydantic ile veri validasyonu
  - HTTPX ile asenkron servisler arası iletişim
  - CORS desteği ile frontend entegrasyonu

Yazar: [Öğrenci Adı]
Tarih: 2024
================================================================================
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import httpx
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
    Müşteri yorumu için veri modeli.
    
    Attributes:
        brand (str): Analiz edilecek marka adı
        text (str): Müşteri yorumu metni
        
    Validators:
        - brand ve text alanları boş olamaz
        - Başta/sonda boşluklar otomatik temizlenir
    """
    brand: str
    text: str

    @field_validator("brand", "text")
    @classmethod
    def not_empty(cls, v: str) -> str:
        """
        Alan validasyonu: Boş string veya sadece boşluk kontrolü.
        
        Args:
            v: Validasyon yapılacak string değer
            
        Returns:
            str: Temizlenmiş (strip edilmiş) string
            
        Raises:
            ValueError: Alan boş veya sadece boşluk içeriyorsa
        """
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


# ============================================================================
# FASTAPI UYGULAMA
# ============================================================================

app = FastAPI(
    title="BrandGuard AI - Ingestion Service",
    description="Müşteri yorumlarını toplayıp Sentiment Analysis servisine yönlendiren API Gateway",
    version="1.0.0"
)

# CORS Middleware: Frontend uygulamalarının API'ye erişmesine izin verir
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production'da belirli origin listesi ile kısıtlanmalı
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sentiment Service URL'i (Docker Compose network üzerinden erişim)
SENTIMENT_SERVICE_URL = "http://sentiment_service:8001/analyze"


# ============================================================================
# API ENDPOINT'LERİ
# ============================================================================


@app.post("/submit")
async def submit_review(review: Review) -> dict:
    """
    Müşteri yorumunu alır, validasyondan geçirir ve Sentiment Analysis
    servisine iletir. Gelen yanıtı olduğu gibi kullanıcıya döner.
    
    İş Akışı:
        1. Gelen veri Pydantic modeli ile otomatik validasyondan geçer
        2. HTTPX ile asenkron olarak Sentiment Service'e POST isteği gönderilir
        3. Sentiment Service'in yanıtı (sentiment, score, timestamp) aynen döndürülür
    
    Args:
        review (Review): Validasyon yapılmış müşteri yorumu modeli
        
    Returns:
        dict: Sentiment Service'den gelen analiz sonucu
            {
                "brand": str,
                "text": str,
                "sentiment": "POSITIVE" | "NEUTRAL" | "CRITICAL",
                "score": float,
                "timestamp": str (ISO 8601)
            }
            
    Raises:
        HTTPException: 
            - 503: Sentiment Service'e bağlanılamazsa
            - 422: Validasyon hatası (Pydantic otomatik döner)
    """
    try:
        logger.info(f"Review received for brand: {review.brand}")
        
        # Sentiment Service'e asenkron HTTP isteği gönder
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                SENTIMENT_SERVICE_URL,
                json=review.model_dump()
            )
            response.raise_for_status()  # HTTP hata kodlarını kontrol et
            
        logger.info(f"Sentiment analysis completed for brand: {review.brand}")
        return response.json()
        
    except httpx.TimeoutException:
        logger.error("Timeout while connecting to Sentiment Service")
        raise HTTPException(
            status_code=503,
            detail="Sentiment Service is not responding. Please try again later."
        )
    except httpx.RequestError as e:
        logger.error(f"Connection error to Sentiment Service: {e}")
        raise HTTPException(
            status_code=503,
            detail="Unable to connect to Sentiment Service. Please check service status."
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"Sentiment Service returned error: {e.response.status_code}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Sentiment Service error: {e.response.text}"
        )


@app.get("/health")
async def health_check() -> dict:
    """
    Servis sağlık kontrolü endpoint'i.
    Docker/Kubernetes gibi orchestrator'lar için kullanılır.
    
    Returns:
        dict: Servis durumu bilgisi
    """
    return {"status": "healthy", "service": "ingestion_service"}


# ============================================================================
# UYGULAMA BAŞLATMA
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Development modunda çalıştırma
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
