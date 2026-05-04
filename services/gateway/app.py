import os, time, json, logging
import httpx                                        # ✅ Fix 1
import pika
from threading import Thread
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO, format="[GATEWAY]%(message)s")
logger = logging.getLogger(__name__)

# configuration
CLASSIFIER_URL = os.environ.get('CLASSIFIER_URL', 'http://classifier:8001')
LOGGER_URL = os.environ.get('LOGGER_URL', 'http://logger:8002')
RABBITMQ_HOST = os.environ.get('RABBITMQ_HOST', 'rabbitmq')
RABBITMQ_PORT = int(os.environ.get('RABBITMQ_PORT', 5672))
RABBITMQ_USER = os.environ.get('RABBITMQ_USER', 'ocr_user')
RABBITMQ_PASSWORD = os.environ.get('RABBITMQ_PASSWORD', 'ocr_password')
EXCHANGE_NAME = 'ocr_exchange'

app = FastAPI(
    title="OCR Gateway Service",
    version="1.0.0",
    description="API Gateway for OCR application. Routes requests to the classifier and logger services."
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

if os.path.exists('static'):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    index_path = os.path.join(os.path.dirname(__file__), 'static', 'index.html')  # ✅ added 'static'
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type='text/html')
    else:
        raise HTTPException(status_code=404, detail="Index page not found.")

rabbit_channel = None

def connect_rabbitmq():
    global rabbit_channel
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASSWORD)
    while True:
        try:
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=RABBITMQ_HOST,
                    port=RABBITMQ_PORT,
                    credentials=credentials,
                    heartbeat=600,
                    connection_attempts=5,
                    retry_delay=5
                )
            )
            rabbit_channel = connection.channel()
            rabbit_channel.exchange_declare(exchange=EXCHANGE_NAME, exchange_type='fanout', durable=True)
            logger.info("Connected to RabbitMQ successfully!")
            break
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}. Retrying in 5 seconds...")
            time.sleep(5)

@app.on_event("startup")
def startup_event():
    Thread(target=connect_rabbitmq, daemon=True).start()

def publish_prediction(prediction_event):
    global rabbit_channel
    if rabbit_channel is None:
        logger.error("Cannot publish prediction: RabbitMQ channel is not available.")
        return
    try:
        rabbit_channel.basic_publish(
            exchange=EXCHANGE_NAME,
            routing_key='',
            body=json.dumps(prediction_event),
            properties=pika.BasicProperties(content_type='application/json', delivery_mode=2)
        )
        logger.info(f"Published prediction event to RabbitMQ: {prediction_event}")
    except Exception as e:
        logger.error(f"Failed to publish prediction event: {e}")

@app.get("/health")
async def health_check():
    async with httpx.AsyncClient() as client:       # ✅ Fix 2
        for name, url in [("Classifier", CLASSIFIER_URL), ("Logger", LOGGER_URL)]:
            try:
                response = await client.get(f"{url}/health")
                if response.status_code == 200:
                    logger.info(f"{name} service is healthy.")
                else:
                    logger.warning(f"{name} service health check returned status code {response.status_code}.")
            except Exception as e:
                logger.error(f"Failed to connect to {name} service at {url}: {e}")
    return {"status": "OK", "message": "OCR Gateway is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    async with httpx.AsyncClient() as client:       # ✅ Fix 3
        try:
            resp = await client.post(
                f"{CLASSIFIER_URL}/predict",
                files={"file": (file.filename, await file.read(), file.content_type)}
            )
            if resp.status_code == 200:
                prediction_result = resp.json()
                logger.info(f"Received prediction from classifier: {prediction_result}")
                publish_prediction({
                    "predicted_text": prediction_result.get("predicted_character"),
                    "confidence": prediction_result.get("confidence"),
                    "image_name": file.filename     # ✅ Fix 4
                })
                return prediction_result
            else:
                logger.error(f"Classifier service returned error status code {resp.status_code}: {resp.text}")
                raise HTTPException(status_code=resp.status_code, detail=f"Classifier error: {resp.text}")
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise HTTPException(status_code=500, detail=str(e))
                                                    # ✅ Fix 5: removed truncated dead code