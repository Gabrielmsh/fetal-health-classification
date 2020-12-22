import numpy as np
import pandas as pd                        #importando todas as bibliotecas utilizadas
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


data_set = pd.read_csv('fetal_health.csv')   #importando a base de dados e salvando em data_set

data_set.isnull().sum()        #verificando se há dados faltantes

data_set.corr()['fetal_health'][:-1].sort_values().plot(kind='bar')     #analisando a correlação dos atributos com as classes
sns.heatmap(data_set.corr())                                        #analisando a correlação entre atributos num heatmap

sns.countplot(x='fetal_health', data=data_set)                    #verificando se as classes estão desbalanceadas


X = data_set.drop('fetal_health', axis=1).values                   #separando os atributos=X das classes=Y
Y = data_set['fetal_health'].values
sm = SMOTE(random_state=13)
X, Y = sm.fit_resample(X, Y)                                  #Utilizando o método SMOTE de oversampling que usa interpolação para gerar instâncias sintéticas
sns.countplot(x=Y)                               #checando se as classes estão balanceadas depois do método de oversampling


parameters= [
  {'C': [1, 10, 20], 'kernel': ['linear']},                             #setando parametros para utilizar no GridSearchCV
  {'C': [1, 10, 20], 'gamma': [0.1, 0.01], 'kernel': ['rbf']},
 ]
svc = svm.SVC()
clf = GridSearchCV(svc, parameters, verbose=1)
clf.fit(X,Y)

clf.best_params_                                       #verificando qual melhor combinação de parâmetros e melhor resultado
clf.best_score_





lb = preprocessing.LabelBinarizer()             #transformando a coluna de classes no formato (n_samples, n_classes) com valores binários
Y = lb.fit_transform(Y)

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.3, random_state=97)           #definição do conjunto de treino e teste

#scaler = MinMaxScaler()
scaler =StandardScaler()
X_train = scaler.fit_transform(X_train)                 #utilizando método de padronização nos atributos
X_test = scaler.transform(X_test)


model = Sequential()
model.add(Dense(21, activation='relu'))                  #definição da rede com modelo sequencial e camadas densas e dropout
model.add(Dropout(0.15))
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=45)    #definindo configurações de parada

model.fit(x=X_train, y=y_train, epochs = 500, validation_data=(X_test,y_test), callbacks=[early_stop])        #treinando o modelo

model_loss = pd.DataFrame(model.history.history)
model_loss.plot()                                         #plotando o histórico do treinando e verificando a perda e se houve overfitting

predic = model.predict_classes(X_test)                      #classificando o conjunto de teste

y_aux = y_test
y_aux = lb.inverse_transform(y_aux)
patologico_total = collections.Counter(y_aux)[3]
suspeito_total = collections.Counter(y_aux)[2]            #declaração de algumas variáveis para calculo das metrica sem a utilização de bibliotecas
normal_total = collections.Counter(y_aux)[1]
y_test=y_test.reshape(4470,1)


tp_normal = 0
tp_suspeito = 0
tp_patologico = 0
fp_normal = 0
fp_suspeito = 0
fp_patologico = 0               #definição de variaveis para o calculo das metricas
fn_normal = 0 
fn_suspeito = 0
fn_patologico = 0                   #tp = true positive, tn = true negative, fp = false positive, fn = false negative
tn_normal = 0
tn_suspeito = 0
tn_patologico = 0


aux = -3
aux2 = -1
for i in predic:                          #logica utilizada para calcular os valores de tp tn fp e fn de cada classe
    aux += 3
    aux2 += 1
    if y_test[aux+i] == 1:
        if i == 0:
            tp_normal += 1
        elif i == 1:
            tp_suspeito += 1
        elif i == 2:
            tp_patologico += 1
    else:
        if y_aux[aux2] == 1:
            fn_normal += 1
        elif y_aux[aux2] == 2:
            fn_suspeito += 1
        elif y_aux[aux2] == 3:
            fn_patologico += 1
        if i == 0:
            fp_normal += 1
        elif i == 1:
            fp_suspeito += 1
        elif i == 2:
            fp_patologico += 1
    if (i!=0) and (y_aux[aux2] != 1):
        tn_normal += 1
    if (i!=1) and (y_aux[aux2] != 2):
        tn_suspeito += 1
    if (i!=2) and (y_aux[aux2] != 3):
        tn_patologico += 1
            
acuracia = (tp_normal+tp_suspeito+tp_patologico)*100/predic.shape[0]      #calculo acuracia


recall_normal = tp_normal/(tp_normal+fn_normal)
recall_suspeito = tp_suspeito/(tp_suspeito+fn_suspeito)       #calculo recall
recall_patologico = tp_patologico/(tp_patologico+fn_patologico)
precision_normal = tp_normal/(tp_normal+fp_normal)
precision_suspeito = tp_suspeito/(tp_suspeito+fp_suspeito)             #calculo precision
precision_patologico = tp_patologico/(tp_patologico+fp_patologico)
f1_score_normal = 2*(precision_normal*recall_normal)/(precision_normal+recall_normal)
f1_score_suspeito = 2*(precision_suspeito*recall_suspeito)/(precision_suspeito+recall_suspeito)            #calculo f1_score
f1_score_patologico = 2*(precision_patologico*recall_patologico)/(precision_patologico+recall_patologico)
specificity_normal = tn_normal/(fp_normal+tn_normal)
specificity_suspeito = tn_suspeito/(fp_suspeito+tn_suspeito)         #calculo specificity
specificity_patologico = tn_patologico/(fp_patologico+tn_patologico)

resultados = pd.DataFrame([[recall_normal,precision_normal,specificity_normal,f1_score_normal],
                           [recall_suspeito,precision_suspeito,specificity_suspeito,f1_score_suspeito],
                           [recall_patologico,precision_patologico,specificity_patologico,f1_score_patologico]],    #disposição dos valores num dataframe para melhor visualização
                          ['Classe 1: normal', 'Classe 2: suspeito', 'Classe 3: patológico'], 
                          ['Recall', 'Precision', 'Specificity', 'F1_score'])

predic = lb.fit_transform(predic)     #transformaçao de algumas variaveis para calculo da curva roc
y_aux = lb.fit_transform(y_aux)

from sklearn.metrics import roc_auc_score           
area_roc_curve = roc_auc_score(y_aux,predic1,multi_class='ovo')          #calculo curva roc

print(acuracia)
print(resultados)                 #impressao dos resultados
print(area_roc_curve)