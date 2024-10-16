from ultralytics import YOLO
import cv2

MODELO = "trained_model.pt"
CAMINHO = "v1.mp4" #insira o caminho do video que deseja

try:
    model = YOLO(MODELO)
    print(f"Modelo carregado com sucesso de {MODELO}")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")

try:
    video_captura = cv2.VideoCapture(CAMINHO)
    
    if not video_captura.isOpened():
        raise FileNotFoundError(f"O vídeo '{CAMINHO}' não foi encontrado ou não pode ser aberto.")
    
    fps = video_captura.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)

    while True:
        ret, frame = video_captura.read()
        
        if not ret:
            print("Fim do vídeo ou erro ao ler o frame.")
            break

        resultado = model.predict(frame)

        frame_bbox = resultado[0].plot()

        cv2.imshow("Detecções EPI", frame_bbox)

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    video_captura.release()
    cv2.destroyAllWindows()
    print("Inferência em vídeo realizada com sucesso.")
    
except Exception as e:
    print(f"Erro durante a inferência: {e}")
