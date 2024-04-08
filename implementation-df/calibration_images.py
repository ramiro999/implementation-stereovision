import cv2

# Inicializa la captura de video para la cámara con índice 0
cap = cv2.VideoCapture(4)
cap2 = cv2.VideoCapture(2)

num = 0

if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

while True:
    # Captura fotograma por fotograma
    ret, img = cap.read()
    ret2, img2 = cap2.read()
    
    # Si la captura fue exitosa, muestra el fotograma
    if ret and ret2:
        cv2.imshow('Camara', img)
        cv2.imshow('Camara2', img2)
        
        # Espera por la tecla 's' para guardar o 'q' para salir
        key = cv2.waitKey(1)
        if key == ord('s'):
            # Guarda el fotograma actual
            #cv2.imwrite('images/stereoLeft/imageL' + str(num) + '.png', img)
            #cv2.imwrite('images/stereoRight/imageR' + str(num) + '.png', img2)
            cv2.imwrite('images/testLeft/imageL' + str(num) + '.png', img)
            cv2.imwrite('images/testRight/imageR' + str(num) + '.png', img2)
            print("Fotos guardadas")
            num += 1
        elif key == ord('q'):
            break
    else:
        print("No se pudo capturar el fotograma")
        break

# Cuando todo está hecho, libera la captura
cap.release()
cv2.destroyAllWindows()
