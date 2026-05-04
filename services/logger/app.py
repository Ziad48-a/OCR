from datetime import datetime
from threading import Thread  
import os ,json,logging,time 
from fastapi import FastAPI
import pika
from fastapi.middleware.cors import CORSMiddleware

#configuration
RABBITMQ_HOST = os.environ.get('RABBITMQ_HOST', 'rabbitmq')
RABBITMQ_PORT = int(os.environ.get('RABBITMQ_PORT', 5672))
RABBITMQ_USER = os.environ.get('RABBITMQ_USER', 'ocr_user')
RABBITMQ_PASSWORD = os.environ.get('RABBITMQ_PASSWORD', 'ocr_password')
QUEUE_NAME = 'ocr_logs'
EXCHANGE_NAME = 'ocr_exchange'

logging.basicConfig(level=logging.INFO, format="[LOGGER]%(message)s") 
logger = logging.getLogger(__name__)

prediction_history = []

app = FastAPI(title="OCR Logger Service",
              version="1.0.0",)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def on_message(channel, method_frame, header_frame, body):
    try :
        event = json.loads(body)
        event["logged_at"] = datetime.now().isoformat()
        event["id"]  = len(prediction_history)
        prediction_history.append(event)

        if len(prediction_history) > 100:
            prediction_history.pop(0)
        
        logger.info(
            f"logged prediction: {event['predicted_text']} for image: {event['image_name']} at {event['logged_at']}"
            f" with confidence: {event['confidence']:.2f}"
        )
        channel.basic_ack(delivery_tag=method_frame.delivery_tag)
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        channel.basic_nack(delivery_tag=method_frame.delivery_tag, requeue=False)


def start_consumer():
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASSWORD)
    try:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(          # ✅ Fix 1: wrap in ConnectionParameters
                host=RABBITMQ_HOST,
                port=RABBITMQ_PORT,
                credentials=credentials,
                heartbeat=600,
                connection_attempts=5,
                retry_delay=5
            )
        )
        channel = connection.channel()

        # ✅ Fix 2: indent these lines inside the try block
        channel.exchange_declare(exchange=EXCHANGE_NAME, exchange_type='fanout', durable=True)
        channel.queue_declare(queue=QUEUE_NAME, durable=True)
        channel.queue_bind(queue=QUEUE_NAME, exchange=EXCHANGE_NAME)

        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue=QUEUE_NAME, on_message_callback=on_message)

        logger.info("Logger service started with rabbitmq, waiting for messages...")
        channel.start_consuming()
    except pika.exceptions.AMQPConnectionError as e:
        logger.error(f"Failed to connect to RabbitMQ: {e}")
        time.sleep(5)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        time.sleep(5)

@app.get("/logs")
def health():
    return {"status": "ok", "message": "Logger service is running"}
from threading import Thread

@app.on_event("startup")
def startup_event():
    Thread(target=start_consumer, daemon=True).start()

@app.get("/history")
def get_history():
    return {
        "total_predictions": len(prediction_history),
        "history": prediction_history
        }
@app.get("/stats")
def get_stats():
    if not prediction_history:
        return {"total_predictions": 0, "average_confidence": 0.0}
    total_predictions = len(prediction_history)
    avg_confidence = sum(event["confidence"] for event in prediction_history) / total_predictions
    return {
        "total_predictions": total_predictions,
        "average_confidence": avg_confidence
    }
