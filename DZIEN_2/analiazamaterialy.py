import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

rng = np.random.default_rng(42)

# 1) Generujemy syntetyczny zbiór danych
n = 400

porowatosc = rng.uniform(5, 80, n)  # %
modul_Younga = rng.lognormal(mean=np.log(3000), sigma=1.0, size=n)  # MPa
degradacja = rng.uniform(0, 30, n)  # % / miesiąc
indeks_bio = rng.uniform(0, 1, n)   # 0-1
kat_kontaktu = rng.uniform(20, 120, n)  # stopnie
chropowatosc = rng.uniform(0.1, 20, n)  # Ra [µm]
przewodnictwo = rng.lognormal(mean=np.log(0.5), sigma=2.0, size=n)  # S/m

powloka_anty = rng.choice([0, 1], size=n, p=[0.6, 0.4])
metoda = rng.choice(['FDM_3D', 'SLM', 'SLS', 'odlewanie', 'elektroprzędzenie'], size=n,
                    p=[0.25, 0.20, 0.20, 0.20, 0.15])
klasa = rng.choice(['ceramika', 'polimer', 'metal', 'kompozyt'], size=n, p=[0.25, 0.35, 0.20, 0.20])

score_kosc = (
    (klasa == 'ceramika').astype(int)*1.0
    + (klasa == 'kompozyt').astype(int)*0.8
    + (porowatosc > 20).astype(int)*0.5
    + (indeks_bio > 0.6).astype(int)*0.7
    + (modul_Younga > 5000).astype(int)*0.4
    - (degradacja > 10).astype(int)*0.6
)

score_stent = (
    (klasa == 'metal').astype(int)*1.0
    + (przewodnictwo > 1.0).astype(int)*0.6
    + (modul_Younga > 10000).astype(int)*0.5
    + (kat_kontaktu < 80).astype(int)*0.2
    - (porowatosc > 40).astype(int)*0.5
)

score_chrzastka = (
    (klasa == 'polimer').astype(int)*0.7
    + (klasa == 'kompozyt').astype(int)*0.6
    + (modul_Younga < 3000).astype(int)*0.6
    + (porowatosc > 40).astype(int)*0.4
    + (indeks_bio > 0.5).astype(int)*0.3
    + (degradacja > 5).astype(int)*0.2
)

score_opatrunek = (
    (powloka_anty == 1).astype(int)*0.9
    + (metoda == 'elektroprzędzenie').astype(int)*0.6
    + (porowatosc > 50).astype(int)*0.4
    + (degradacja > 8).astype(int)*0.3
    + (kat_kontaktu < 70).astype(int)*0.2
)

score_nosnik = (
    (klasa == 'polimer').astype(int)*0.5
    + (porowatosc > 60).astype(int)*0.4
    + (degradacja > 10).astype(int)*0.6
    + (indeks_bio > 0.4).astype(int)*0.3
    + (metoda == 'SLS').astype(int)*0.2
)

scores = np.vstack([score_kosc, score_stent, score_chrzastka, score_opatrunek, score_nosnik]).T
labels_list = np.array(['implant_kostny', 'stent_naczyniowy', 'chrzastka_stawowa', 'opatrunek_antybakteryjny', 'nosnik_leku'])
y = labels_list[np.argmax(scores + rng.normal(0, 0.1, size=scores.shape), axis=1)]

df = pd.DataFrame({
    'porowatosc_%': np.round(porowatosc, 2),
    'modul_Younga_MPa': np.round(modul_Younga, 2),
    'degradacja_%_mies': np.round(degradacja, 2),
    'indeks_bioaktywnosci': np.round(indeks_bio, 3),
    'kat_kontaktu_deg': np.round(kat_kontaktu, 1),
    'chropowatosc_Ra_um': np.round(chropowatosc, 2),
    'przewodnictwo_S_m': np.round(przewodnictwo, 5),
    'powloka_antybakteryjna': powloka_anty,
    'metoda_produkcji': metoda,
    'klasa_materialu': klasa,
    'zastosowanie_docelowe': y
})

csv_path = '/mnt/data/biomaterials_dataset.csv'
df.to_csv(csv_path, index=False)

# Model SVM
X = df.drop(columns=['zastosowanie_docelowe'])
y = df['zastosowanie_docelowe']

num_cols = ['porowatosc_%', 'modul_Younga_MPa', 'degradacja_%_mies', 'indeks_bioaktywnosci',
            'kat_kontaktu_deg', 'chropowatosc_Ra_um', 'przewodnictwo_S_m']
cat_cols = ['powloka_antybakteryjna', 'metoda_produkcji', 'klasa_materialu']

pre = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)

svm_clf = Pipeline(steps=[
    ('pre', pre),
    ('svm', SVC(kernel='rbf', C=3.0, gamma='scale', probability=True, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
svm_clf.fit(X_train, y_train)

y_pred = svm_clf.predict(X_test)

report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=labels_list)

import caas_jupyter_tools
caas_jupyter_tools.display_dataframe_to_user(name="Podgląd danych biomedycznych (pierwsze 20 wierszy)", dataframe=df.head(20))

print("Ścieżka do pliku CSV:", csv_path)
print("\nRaport klasyfikacji (SVM RBF):")
print(report)
print("\nMacierz pomyłek (kolejność etykiet):", list(labels_list))
print(cm)
