from ultralytics import YOLO

DATASET_URL = "https://universe.roboflow.com/ds/lw9kCKoxSY?key=F2z1cc7nxN"
MODEL_NAME = "yolov8m.pt"
IMG_SIZE = 640
EPOCHS = 20
MODEL_SAVE_PATH = "modelo_treinado.pt"

# Load the pre-trained YOLOv8 model
model = YOLO(MODEL_NAME)

# Train the model
try:
    model.train(data=DATASET_URL, epochs=EPOCHS, imgsz=IMG_SIZE)
except Exception as e:
    print(f"Erro durante o treinamento: {e}")

# Evaluate the model after training
try:
    metrics = model.val()
    print("Métricas de avaliação:", metrics)
except Exception as e:
    print(f"Erro durante a avaliação: {e}")

# Save the trained model
try:
    model.save(MODEL_SAVE_PATH)
    print(f"Modelo treinado salvo com sucesso em {MODEL_SAVE_PATH}")
except Exception as e:
    print(f"Erro durante o salvamento do modelo: {e}")

# Export the trained model to different formats if necessary
try:
    model.export(format="onnx")  # You can export to onnx, torchscript, coreml, etc.
    print("Modelo exportado com sucesso.")
except Exception as e:
    print(f"Erro durante a exportação do modelo: {e}")