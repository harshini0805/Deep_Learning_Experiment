"""Ex 6: LSTM vs GRU on identical frozen MobileNetV2 features, repeated over 10 stratified 25/7/8 splits (paired)."""
import os, sys, json, glob, time
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
U = '/mnt/user-data/uploads/Training/checkpoints/video_features'
CL = ["Basketball", "Biking", "WalkingWithDog", "JumpingJack", "TennisSwing"]
X, y = [], []
for i, c in enumerate(CL):
    for f in sorted(glob.glob(f'{U}/{c}__v_*.npy'))[:8]: X.append(np.load(f)); y.append(i)
X = np.stack(X).astype('float32'); y = np.array(y); assert X.shape == (40, 10, 1280)
def build(cls):
    i = layers.Input(shape=(10, 1280)); x = cls(32, name='recurrent')(i); x = layers.Dropout(0.2)(x)
    o = layers.Dense(5, activation='softmax', dtype='float32')(x)
    m = keras.Model(i, o); m.compile(optimizer=optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy']); return m
res = []
for s in range(10):
    Xtv, Xte, ytv, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=s)
    Xtr, Xva, ytr, yva = train_test_split(Xtv, ytv, test_size=0.2, stratify=ytv, random_state=s)
    row = dict(split=s)
    for nm, cls in [('lstm', layers.LSTM), ('gru', layers.GRU)]:
        tf.keras.utils.set_random_seed(s); m = build(cls)
        h = m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=20, batch_size=32, verbose=0)
        yp = m.predict(Xte, verbose=0).argmax(1)
        row[nm] = dict(acc=accuracy_score(yte, yp) * 100, f1=f1_score(yte, yp, average='macro', zero_division=0) * 100,
                       val_acc=float(h.history['val_accuracy'][-1]) * 100, train_acc=float(h.history['accuracy'][-1]) * 100, n_test=len(yte))
    res.append(row); print('VIDEO split', s, round(row['lstm']['acc'], 1), round(row['gru']['acc'], 1), flush=True)
json.dump(res, open('/tmp/w/ex/results/video_cv.json', 'w')); print('ALL_VIDEO_DONE', flush=True)
