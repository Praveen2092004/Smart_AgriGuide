import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

CROP_PROFILES = {
    'rice':        {'N': (60, 100),  'P': (35, 60),  'K': (35, 45),  't': (20, 27), 'h': (80, 85), 'ph': (5.0, 7.0), 'r': (180, 300)},
    'maize':       {'N': (60, 100),  'P': (35, 60),  'K': (15, 25),  't': (18, 27), 'h': (55, 75), 'ph': (5.5, 7.0), 'r': (60, 110)},
    'chickpea':    {'N': (20, 60),   'P': (55, 80),  'K': (75, 85),  't': (17, 20), 'h': (14, 20), 'ph': (6.0, 8.5), 'r': (65, 95)},
    'kidneybeans': {'N': (15, 40),   'P': (55, 80),  'K': (15, 25),  't': (15, 25), 'h': (18, 25), 'ph': (5.5, 6.0), 'r': (60, 150)},
    'pigeonpeas':  {'N': (15, 40),   'P': (55, 75),  'K': (15, 25),  't': (24, 38), 'h': (30, 68), 'ph': (4.5, 7.5), 'r': (90, 200)},
    'mothbeans':   {'N': (10, 35),   'P': (35, 60),  'K': (15, 25),  't': (24, 32), 'h': (40, 65), 'ph': (3.5, 9.5), 'r': (30, 75)},
    'mungbean':    {'N': (10, 40),   'P': (35, 60),  'K': (15, 25),  't': (27, 30), 'h': (80, 90), 'ph': (6.2, 7.2), 'r': (35, 60)},
    'blackgram':   {'N': (30, 60),   'P': (55, 80),  'K': (15, 25),  't': (25, 35), 'h': (60, 70), 'ph': (6.5, 7.5), 'r': (60, 75)},
    'lentil':      {'N': (15, 40),   'P': (55, 80),  'K': (15, 25),  't': (18, 30), 'h': (60, 70), 'ph': (6.0, 7.5), 'r': (35, 55)},
    'pomegranate': {'N': (15, 40),   'P': (10, 30),  'K': (35, 45),  't': (18, 25), 'h': (85, 95), 'ph': (5.5, 7.2), 'r': (100, 115)},
    'banana':      {'N': (80, 120),  'P': (70, 95),  'K': (45, 55),  't': (25, 30), 'h': (75, 85), 'ph': (5.5, 6.5), 'r': (90, 120)},
    'mango':       {'N': (10, 35),   'P': (15, 35),  'K': (25, 35),  't': (27, 36), 'h': (45, 55), 'ph': (4.5, 7.0), 'r': (85, 105)},
    'grapes':      {'N': (10, 35),   'P': (120, 145),'K': (195, 205),'t': (8, 42),  'h': (80, 85), 'ph': (5.5, 6.5), 'r': (65, 75)},
    'watermelon':  {'N': (80, 120),  'P': (10, 30),  'K': (45, 55),  't': (24, 27), 'h': (80, 90), 'ph': (6.0, 7.0), 'r': (40, 60)},
    'muskmelon':   {'N': (80, 120),  'P': (10, 30),  'K': (45, 55),  't': (27, 30), 'h': (90, 95), 'ph': (6.0, 6.8), 'r': (20, 30)},
    'apple':       {'N': (15, 40),   'P': (120, 145),'K': (195, 205),'t': (21, 24), 'h': (90, 95), 'ph': (5.5, 6.5), 'r': (100, 125)},
    'orange':      {'N': (10, 40),   'P': (10, 30),  'K': (5, 15),   't': (10, 35), 'h': (90, 95), 'ph': (6.0, 8.0), 'r': (100, 120)},
    'papaya':      {'N': (30, 70),   'P': (45, 70),  'K': (45, 55),  't': (23, 44), 'h': (90, 95), 'ph': (6.5, 7.0), 'r': (40, 250)},
    'coconut':     {'N': (15, 40),   'P': (10, 30),  'K': (25, 35),  't': (25, 29), 'h': (90, 99), 'ph': (5.5, 6.5), 'r': (130, 225)},
    'cotton':      {'N': (100, 140), 'P': (35, 60),  'K': (15, 25),  't': (22, 26), 'h': (75, 85), 'ph': (6.0, 8.0), 'r': (60, 100)},
    'jute':        {'N': (60, 95),   'P': (35, 60),  'K': (35, 45),  't': (23, 26), 'h': (70, 90), 'ph': (6.0, 7.5), 'r': (150, 200)},
    'coffee':      {'N': (80, 120),  'P': (15, 35),  'K': (25, 35),  't': (23, 28), 'h': (50, 70), 'ph': (6.0, 7.5), 'r': (115, 200)}
}

CSV_FILENAME = "Crop_recommendation.csv"

if not os.path.exists(CSV_FILENAME):
    np.random.seed(42)
    rows = []
    for crop, bounds in CROP_PROFILES.items():
        for _ in range(100):
            n = np.random.uniform(bounds['N'][0], bounds['N'][1])
            p = np.random.uniform(bounds['P'][0], bounds['P'][1])
            k = np.random.uniform(bounds['K'][0], bounds['K'][1])
            t = np.random.uniform(bounds['t'][0], bounds['t'][1])
            h = np.random.uniform(bounds['h'][0], bounds['h'][1])
            ph = np.random.uniform(bounds['ph'][0], bounds['ph'][1])
            r = np.random.uniform(bounds['r'][0], bounds['r'][1])
            rows.append([round(n, 2), round(p, 2), round(k, 2), round(t, 2), round(h, 2), round(ph, 2), round(r, 2), crop])
    df = pd.DataFrame(rows, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label'])
    df.to_csv(CSV_FILENAME, index=False)
else:
    df = pd.read_csv(CSV_FILENAME, encoding='utf-8-sig')

df.columns = df.columns.str.strip()
col_map = {}
for col in df.columns:
    clean = col.lower().replace('\ufeff', '')
    if clean in ['n', 'nitrogen']: col_map[col] = 'N'
    elif clean in ['p', 'phosphorus', 'phosphorous']: col_map[col] = 'P'
    elif clean in ['k', 'potassium']: col_map[col] = 'K'
    elif clean in ['temperature', 'temp']: col_map[col] = 'temperature'
    elif clean in ['humidity', 'relative_humidity']: col_map[col] = 'humidity'
    elif clean in ['ph', 'ph_value']: col_map[col] = 'ph'
    elif clean in ['rainfall', 'rain']: col_map[col] = 'rainfall'
    elif clean in ['label', 'crop', 'target']: col_map[col] = 'label'

df.rename(columns=col_map, inplace=True)
df = df.dropna()

X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=16)
rf.fit(X_train, y_train)

acc = accuracy_score(y_test, rf.predict(X_test))
print(f"Training Complete! Stratified Test Accuracy: {acc * 100:.2f}%")
joblib.dump(rf, "crop_recommender.joblib")
print("Saved production artifact: crop_recommender.joblib")