# ReconocimientoFacial

### Autores ✒️

_Todos los colaboradores del proyecto desde sus inicios son:_

- [Miguel Muzo](https://github.com/Miguel-EMC)
- [Elena Peréz](https://github.com/kevinpinan)

--- 
## Video de Youtube 📌

En el siguiente enlace se podra visualizar todos los procedimiento que se realizo [Video](https://youtu.be/avcwnAtKPWU)

### Descripción 📄
El equipo de trabajo para el proyecto denominado, es la creacion de un programa para la deteccion de emociones faciales.

![image](https://github.com/Miguel-EMC/ReconocimientoFacial/blob/master/Images/Screenshot%202022-09-02%20161525.png)

---
## Content 🚀
- [HERRAMIENTAS UTILIZADAS](#HERRAMIENTAS-UTILIZADAS)
	- [KERAS](#KERAS)
	- [MATPLOTLIB](#MATPLOTLIB)
	- [SKLEARN](#SKLEARN)
	- [VISUAL-STUDIO](#VISUAL-STUDIO)
- [DESARROLLO DEL PROYECTO](#DESARROLLO-DEL-PROYECTO)

## HERRAMIENTAS-UTILIZADAS 📋
_Para poder desarrollar el proyecto es necesario instalar liberias de python_

### KERAS
Es una librería de código abierto (con licencia MIT) escrita en Python para acelerar la creación de redes neuronales para ello, Keras no funciona como un framework independiente, sino como una interfaz de uso intuitivo (API) que permite acceder a varios frameworks de aprendizaje automático y desarrollarlos.
### MATPLOTLIB
Es una librería de Python especializada en la creación de gráficos en dos dimensiones, también a partir de datos contenidos en listas o arrays en el lenguaje de programación Python y su extensión matemática NumPy. Proporciona una API, pylab, diseñada para recordar a la de MATLAB. Matplotlib se basa en varios elementos clave. Una “figura” es una ilustración completa. Cada trazado de esa figura se llama “eje”.
### SKLEARN
Es una de estas librerías gratuitas para Python. Cuenta con algoritmos de clasificación, regresión, clustering y reducción de dimensionalidad. Además, presenta la compatibilidad con otras librerías de Python como NumPy, SciPy y matplotlib, es la librería más útil para Machine Learning en Python, es de código abierto y es reutilizable en varios contextos, fomentando el uso académico y comercial. Proporciona una gama de algoritmos de aprendizaje supervisados y no supervisados en Python.
### VISUAL-STUDIO
Visual Studio Code (VS Code) es un editor de código fuente desarrollado por Microsoft. Es software libre y multiplataforma, está disponible para Windows, GNU/Linux y macOS. VS Code tiene una buena integración con Git, cuenta con soporte para depuración de código, y dispone de un sinnúmero de extensiones, que básicamente te da la posibilidad de escribir y ejecutar código en cualquier lenguaje de programación.

## DESARROLLO-DEL-PROYECTO
Se instaló previamente las librerías de keras, matpltlib, numpy, tensorflow, ya que se utilizara para poder realizar el proyecto, luego se llamó a las librerías dentro del código del proyecto.

```py
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
import os
from matplotlib import pyplot as plt
import numpy as np
```
Después se crea un tamaño para las imagenes que se van a mostrar dentro del programa, y para poder leer la dataset se crean las variables.

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

Se utiliza la librería de keras, la cual cuenta con la clase ImageDataGenerator, la cual va ayudar a generar bloques o batch, esto se realiza porque cuando se requiere entrenar modelos con resoluciones mayores, por lo que se requiere mucha memoria, por lo tanto se necesita dividir el proceso de entrenamiento en bloques de menor tamaño de imágenes, por eso se utiliza la librería keras, en la que se realiza la técnica llamada data augmentation.

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
Se verifica el generador trazando las emociones faciales de las imágenes generadas aleatoriamente, para ello se crea un arreglo denominado class_labels, en la que se debe incluir, las 7 emociones, por ejemplo: feliz, triste, nervioso, enojado, neutral, sorprendido y finalmente se imprime una función random las imágenes.
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

Finalmente se construye, para la matriz confusión con ayuda de la librería tensorflow con el metodo sklearn.metrics.
```py
#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, predictions)
print(cm)
import seaborn as sns
sns.heatmap(cm, annot=True)
```
