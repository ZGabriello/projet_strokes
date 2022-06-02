import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pickle

df = pd.read_csv('strokes.csv', index_col= 0)

def mult(age):
  return age * 100

df.loc[df.age<1, 'age']= df.age.apply(mult)

# on binarise gender Male = 1 et Female = 0
df['gender'] = df['gender'].replace(['Male', 'Female'], [1, 0])

# on binarise ever_married Yes = 1 et No = 0
df['ever_married'] = df['ever_married'].replace(['Yes', 'No'], [1, 0])

# on binarise residence_type Rural = 1 et Urban = 0
df['Residence_type'] = df['Residence_type'].replace(['Rural', 'Urban'], [1, 0])

# on binarise smoking_status never smoked et Unknown = 1 et smokes et formerly smoked = 0
df['smoking_status'] = df['smoking_status'].replace(['never smoked', 'Unknown', 'formerly smoked', 'smokes'], [1, 1, 0, 0])

# on binarise work_type Govt_job, Private, Self-employed = 1 et Never_worked, children = 0 (on mets 1 pour ceux qui travaillent et 0 pour ceux qui n'ont jamais travaillé)
df['work_type'] = df['work_type'].replace(['Govt_job', 'Private', 'Self-employed', 'Never_worked', 'children'], [1, 1, 1, 0, 0])

#on prend l'index de la ligne où genre est égale à other
indexOther = df[ df['gender'] == 'Other' ].index

# on supprime ces lignes
df.drop(indexOther , inplace=True)

#on change le type de gender, ever_married et residence_type en int
df['gender'] = df['gender'].astype(int)
df['ever_married'] = df['ever_married'].astype(int)
df['Residence_type'] = df['Residence_type'].astype(int)
df['smoking_status'] = df['Residence_type'].astype(int)
df['work_type'] = df['Residence_type'].astype(int)

# on supprime les valeurs manquantes de bmi
# en effet le nombre de valeur est très peu significatif par rapport au contenu total du dataframe , de plus la corrélation entre stroke et bmi est faible donc la supprésion de ces Nans n'est pas très désavantageux
df = df.dropna(subset=['bmi'])
print("Le nombre de valeurs manquantes pour chaque colonnes : \n", df.isna().sum())

#on supprime les variables inutiles comme ever_married, Residence_type et work_type
df = df.drop(["Residence_type"], axis = 1)

# Séparation des variables explicatives de df dans un Dataframe X et la variable cible dans une Series y
X = df.drop("stroke", axis = 1)

y = df['stroke']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30, random_state=42,stratify=y)

# On affiche les dimensions des datasets après avoir appliquer la fonction

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

# On instantie le modèle et on l'entraîne

model_log=LogisticRegression(solver="newton-cg").fit(X_train,y_train)

# On prédit les y à partir de X_test

y_pred=model_log.predict(X_test)

# On affiche les coefficients obtenus

coeff=model_log.coef_

# On affiche la constante

intercept=model_log.intercept_

# On calcule les odd ratios

odd_ratios=np.exp(model_log.coef_)

# On crée un dataframe qui combine à la fois variables, coefficients et odd-ratios

resultats=pd.DataFrame(df.drop("stroke", axis = 1).columns, columns=["Variables"])

resultats['Coefficients']=model_log.coef_.tolist()[0]

resultats['Odd_Ratios']=np.exp(model_log.coef_).tolist()[0]

# On choisit d'afficher les variables avec l'odd ratio le plus élevé et le plus faible

resultats


# application de la regression logistique
output = open("output.pickle", "wb")
pickle.dump(resultats, output)
output.close()

# On calcule la matrice de confusion et on l'affiche

print("\n Matrice de confusion \n", confusion_matrix(y_test,y_pred))

# On peut également afficher la signification de chaque élément de la matrice lorsque l'on a une problématique binaire

tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()

print("\n Vrais négatifs:",tn,"\n Faux positifs:",fp,"\n Faux négatifs:",fn,"\n Vrais positifs:",tp)

dictionnaire = {
    "vrai_negatif": tn,
    "faux_positif": fp,
    "faux_negatif": fn,
    "vrai_positif": tp
}

print(dictionnaire)

# performance
perf = open("perf.pickle", "wb")
pickle.dump(dictionnaire, perf)
perf.close()
