# Detecci贸n Reconocimiento Facial

### Autores 鉁掞笍

_Todos los colaboradores del proyecto desde sus inicios son:_

- [Miguel Muzo](https://github.com/Miguel-EMC)
- [Elena Per茅z](https://github.com/kevinpinan)

--- 
## Video de Youtube 馃搶

En el siguiente enlace se podra visualizar todos los procedimiento que se realizo [Video](https://youtu.be/avcwnAtKPWU)

### Descripci贸n 馃搫
El equipo de trabajo para el proyecto denominado, es la creacion de un programa para la deteccion de emociones faciales.

![image](https://github.com/Miguel-EMC/ReconocimientoFacial/blob/master/Images/Screenshot%202022-09-02%20161525.png)

---
## Content 馃殌
- [HERRAMIENTAS UTILIZADAS](#HERRAMIENTAS-UTILIZADAS)
	- [KERAS](#KERAS)
	- [MATPLOTLIB](#MATPLOTLIB)
	- [SKLEARN](#SKLEARN)
	- [VISUAL-STUDIO](#VISUAL-STUDIO)
- [DESARROLLO DEL PROYECTO](#DESARROLLO-DEL-PROYECTO)

## HERRAMIENTAS-UTILIZADAS 馃搵
_Para poder desarrollar el proyecto es necesario instalar liberias de python_

### KERAS
Es una librer铆a de c贸digo abierto (con licencia MIT) escrita en Python para acelerar la creaci贸n de redes neuronales para ello, Keras no funciona como un framework independiente, sino como una interfaz de uso intuitivo (API) que permite acceder a varios frameworks de aprendizaje autom谩tico y desarrollarlos.
### MATPLOTLIB
Es una librer铆a de Python especializada en la creaci贸n de gr谩ficos en dos dimensiones, tambi茅n a partir de datos contenidos en listas o arrays en el lenguaje de programaci贸n Python y su extensi贸n matem谩tica NumPy. Proporciona una API, pylab, dise帽ada para recordar a la de MATLAB. Matplotlib se basa en varios elementos clave. Una 鈥渇igura鈥? es una ilustraci贸n completa. Cada trazado de esa figura se llama 鈥渆je鈥?.
### SKLEARN
Es una de estas librer铆as gratuitas para Python. Cuenta con algoritmos de clasificaci贸n, regresi贸n, clustering y reducci贸n de dimensionalidad. Adem谩s, presenta la compatibilidad con otras librer铆as de Python como NumPy, SciPy y matplotlib, es la librer铆a m谩s 煤til para Machine Learning en Python, es de c贸digo abierto y es reutilizable en varios contextos, fomentando el uso acad茅mico y comercial. Proporciona una gama de algoritmos de aprendizaje supervisados y no supervisados en Python.
### VISUAL-STUDIO
Visual Studio Code (VS Code) es un editor de c贸digo fuente desarrollado por Microsoft. Es software libre y multiplataforma, est谩 disponible para Windows, GNU/Linux y macOS. VS Code tiene una buena integraci贸n con Git, cuenta con soporte para depuraci贸n de c贸digo, y dispone de un sinn煤mero de extensiones, que b谩sicamente te da la posibilidad de escribir y ejecutar c贸digo en cualquier lenguaje de programaci贸n.

## DESARROLLO-DEL-PROYECTO
Se instal贸 previamente las librer铆as de keras, matpltlib, numpy, tensorflow, ya que se utilizara para poder realizar el proyecto, luego se llam贸 a las librer铆as dentro del c贸digo del proyecto.

```py
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
import os
from matplotlib import pyplot as plt
import numpy as np
```
Despu茅s se crea un tama帽o para las imagenes que se van a mostrar dentro del programa, y para poder leer la dataset se crean las variables.

```py
#image size
IMG_HEIGHT=48 
IMG_WIDTH = 48
batch_size=32
```
```py
train_data_dir='./train/'
validation_data_dir='./test/'
```

Se utiliza la librer铆a de keras, la cual cuenta con la clase ImageDataGenerator, la cual va ayudar a generar bloques o batch, esto se realiza porque cuando se requiere entrenar modelos con resoluciones mayores, por lo que se requiere mucha memoria, por lo tanto se necesita dividir el proceso de entrenamiento en bloques de menor tama帽o de im谩genes, por eso se utiliza la librer铆a keras, en la que se realiza la t茅cnica llamada data augmentation.

```py
train_datagen = ImageDataGenerator(
					rescale=1./255,
					rotation_range=30,
					shear_range=0.3,
					zoom_range=0.3,
					horizontal_flip=True,
					fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
					train_data_dir,
					color_mode='grayscale',
					target_size=(IMG_HEIGHT, IMG_WIDTH),
					batch_size=batch_size,
					class_mode='categorical',
					shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
							validation_data_dir,
							color_mode='grayscale',
							target_size=(IMG_HEIGHT, IMG_WIDTH),
							batch_size=batch_size,
							class_mode='categorical',
							shuffle=True)
```
Se verifica el generador trazando las emociones faciales de las im谩genes generadas aleatoriamente, para ello se crea un arreglo denominado class_labels, en la que se debe incluir, las 7 emociones, por ejemplo: feliz, triste, nervioso, enojado, neutral, sorprendido y finalmente se imprime una funci贸n random las im谩genes.
```py
class_labels=['Angry','Disgust', 'Fear', 'Happy','Neutral','Sad','Surprise']

img, label = train_generator.__next__()

import random

i=random.randint(0, (img.shape[0])-1)
image = img[i]
labl = class_labels[label[i].argmax()]
plt.imshow(image[:,:,0], cmap='gray')
plt.title(labl)
plt.show()
```
![image](https://github.com/Miguel-EMC/ReconocimientoFacial/blob/master/Images/output.png)

Finalmente se construye, para la matriz confusi贸n con ayuda de la librer铆a tensorflow con el metodo sklearn.metrics.
```py
#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, predictions)
print(cm)
import seaborn as sns
sns.heatmap(cm, annot=True)
```
![image](https://github.com/Miguel-EMC/ReconocimientoFacial/blob/master/Images/Screenshot%202022-09-02%20165437.png)

Crear la cascada para poder graficar la caja delimitadora que se va a encontrar alrededor de la cara, de  igual manera se incluye funciones para poder detectar los rostros de las personas dentro del video.

```py
while True:
  # Encuentra la cascada haar para dibujar la caja delimitadora alrededor de la cara
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detectar los rostros disponibles en la c谩mara
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # tomar cada rostro disponible en la c谩mara y preprocesarlo
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
Se reproduce el video y se comienza el programa a detectar las emociones faciales. 
![image](https://github.com/Miguel-EMC/ReconocimientoFacial/blob/master/Images/Screenshot%202022-09-02%20165846.png)
