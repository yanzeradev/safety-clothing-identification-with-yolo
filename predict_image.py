from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

MODELO = "trained_model.pt"
CAMINHO_IMAGEM = "img1.jpg" #insira o caminho da foto que deseja

try:
    model = YOLO(MODELO)
    print(f"Modelo carregado com sucesso de {MODELO}")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")

try:
    imagem = cv2.imread(CAMINHO_IMAGEM)

    if imagem is None:
        raise FileNotFoundError(f"A imagem '{CAMINHO_IMAGEM}' não foi encontrada ou não pode ser aberta.")

    resultado = model.predict(imagem)
    imagem_bbox = resultado[0].plot()

    imagem_rgb = cv2.cvtColor(imagem_bbox, cv2.COLOR_BGR2RGB)

    plt.imshow(imagem_rgb)
    plt.axis('off')
    plt.show()

    print("Inferência em imagem realizada com sucesso.")

except Exception as e:
    print(f"Erro durante a inferência: {e}")
