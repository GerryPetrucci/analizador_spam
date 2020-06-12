import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import re
import matplotlib.pyplot as plt
plt.style.use('ggplot')
#%matplotlib inline


def porcentajeClases(output):
    percent = output.value_counts() 
    clase1 = (percent[0]*100)/(percent[0]+percent[1])
    clase2 = (percent[1]*100)/(percent[0]+percent[1])
    return (clase1, clase2)

#Se importan los datos

sms = pd.read_csv("C:\\Users\\GerardoTavares\\OneDrive - Norcul S.A. de C.V\\Escritorio\\Analizador de SPAM\\Analizador de SPAM\\sms_spam.csv")
sms.head()

#Exploración y preprocesamiento de los datos

print('Tamaño del dataset')
print('Número de ejemplos: {}'.format(sms.shape[0]))
print('Número de características: {}'.format(sms.shape[1]))

#Balance de los datos(target) en el dataset

print('Número de mensajes importantes y spam: ')
print(sms['type'].value_counts())


[nosp, sp] = porcentajeClases(sms['type'])
print('Porcentaje de mensajes importantes: {}%'.format(nosp))
print('Porcentaje de mensajes spam: {}%'.format(sp))

#Limpieza de los textos

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
#Se convierten los mensajes a minúsculas
sms['text'] = sms.text.map(lambda x: x.lower())
#Se quitan los signos de puntuación
sms['text'] = sms.text.str.replace('[^\w\s]','')
#Se eliminan los números
sms['text'] = [re.sub('[0-9]', '', i) for i in sms['text']]
#Se hace la tokenización
sms['text'] = sms['text'].apply(nltk.word_tokenize)
nltk_words = list(stopwords.words('english'))
#Lista de "stopwords" en inglés
for i in range(0,len(sms)):
    sms['text'][i] = [w for w in sms['text'][i] if w not in nltk_words]
#Se convierte la lista de palabras en un string separado por espacios
sms['text'] = sms['text'].apply(lambda x: ' '.join(x))

#Se verifican los cambios
sms.head()

#El siguiente apartado se creó para ir monitoreando las pruebas
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
def generatingWordCloud(text):
    wordcloud = WordCloud(width=800, height=800,background_color='white', min_font_size=10).generate(str(text))  
    plt.figure(figsize=(6, 6), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
#Se visualizan los datos
print('Wordcloud de mensajes importantes')
generatingWordCloud(sms.loc[sms['type'] == 'ham'].text)
print('Wordcloud de mensajes spam')
generatingWordCloud(sms.loc[sms['type'] == 'spam'].text)

#Creación del espacio vectorial
from sklearn.feature_extraction.text import  CountVectorizer
count_vect = CountVectorizer() #Creamos elmodelo de espacio vectorial (bolsa de palabras)  con los valores tf
bagofWords = count_vect.fit_transform(sms['text'])
print('Tamaño del modelo de espacio vectorial con todas las palabras: ')
print(bagofWords.shape)

#Tamaño del modelo de espacio vectorial
count_vect = CountVectorizer(min_df = 5)
bagofWords = count_vect.fit_transform(sms['text'])
print('\nTamaño del modelo de espacio vectorial con las palabras que aparecen en más de 5 mensajes: ')
(bagofWords.shape)

from scipy.sparse import csr_matrix
from scipy import sparse
bagofWords = bagofWords.toarray()
bagofWords = csr_matrix(bagofWords)

#Separación del dataset en un conjunto de entrenamiento y de prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(bagofWords, sms['type'], test_size=0.2, random_state=42)

[nosp, sp] = porcentajeClases(y_train)
print('Porcentaje de mensajes importantes en el conjunto de entrenamiento: {}%'.format(nosp))
print('Porcentaje de mensajes spam en el conjunto de entrenamiento: {}%'.format(sp))

[nosp, sp] = porcentajeClases(y_test)
print('Porcentaje de mensajes importantes en el conjunto de prueba: {}%'.format(nosp))
print('Porcentaje de mensajes spam en el conjunto de prueba: {}%'.format(sp))

#Creación del modelo
from sklearn.naive_bayes import MultinomialNB
# The multinomial Naive Bayes classifier is suitable for classification with discrete features (words)
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

#Evaluación del modelo y Precisión

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
y_pred = mnb.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred)
print('Precisión en el conjunto de entrenamiento: {}'.format(train_accuracy))
y_pred = mnb.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print('Precisión en el conjunto de prueba: {}'.format(test_accuracy))

#Matriz de confsión
print(pd.crosstab(y_test,y_pred,rownames=['True'],colnames=['Predicted'],margins=True))

#Reporte de clasificación
print(classification_report(y_test,y_pred))
