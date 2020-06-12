# analizador_spam
Backend del desarrollo de un modelo para clasificar correos en inglés como spam o no.

# Backend Proyecto Clasificador Binario de Correos
# Orozco Magadán Brandon
# Tavares Rosales Gerardo Daniel

Este aplicativo tiene la funcion de entrenar un modelo basado en NaiveBayes para etiquetar un correo en "spam" o "no spam"
predefinidos en el archivo [sms_spam.csv].
El presente desarrollo sólo cuenta con una backend, por lo que posteriormente se buscará realizarle una interfaz para que el usuario pueda introducir
su correo y así una vez entrenado el modelo, verifique si su correo es spam o no.

## Getting Started

Como se mencionó, el código existente es del backend por lo que el usuario sólo podrá correr el código en una máquina que tenga Anaconda o Miniconda.

### Sobre el algoritmo

El algoritmo es muy sencillo, obtiene el dataset y se limpia de signos de puntuación, posteriormente se tokeniza y se genera el bagofwords para trabajar.
El código tiene una pequeña parte para hacer un word cloud de cada categoría (spam y no spam) y mostrarle al usuario de una manera gráfica las
palabras que más se repiten en cada categoría.
Al final se entrena y evalúa el modelo para saber su exactitud.


### TRAINING

El programa utiliza varios parámetros para el training, también se elije el tamaño de la prueba (en porcentaje del dataset)

#### Eligiendo el modelo

Estos son los modelos que se pueden elegir basados en NaiveBayes con sus % de precisión promedio

```
1: GaussianNB(), # .34 Accuracy
2: MultinomialNB(), # .57 Accurac
3: BernoulliNB(), #.47 Accuracy
4: ComplementNB(), # .64 Accuracy -------------- Se usa por default si no se asigna un valor a model
```

### PREDICCIONES

Por el momento el desarrollo actual no cuenta con predicciones, ya que aún no tiene interfaz para poder implementar

## Built With


* [Sklearn](https://scikit-learn.org) - Para los modelos
* [NLTK](https://www.nltk.org/) - Para algunas Corpora como [stopwords]
* [WordCloud](https://pypi.org/project/wordcloud/) - Para poder obtener las imágenes de las palabras más repetidas
* [Pandas](https://pandas.pydata.org/) - Generación de DataFrames y manejo de  los datos


## Autor

* **Orozco Magadán Brandon, Tavares Rosales Gerardo Daniel** - *Initial work* - gerrypetrucci(https://github.com/gerrypetrucci)

