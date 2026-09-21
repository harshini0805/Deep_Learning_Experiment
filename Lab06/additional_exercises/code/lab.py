"""Re-implementation of the notebook pipeline (identical preprocessing / models / training protocol)."""
import os, json, time, glob
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

U = '/mnt/user-data/uploads/Training/data/UCI_HAR_Dataset/UCI HAR Dataset'
SIGNALS = ["body_acc_x","body_acc_y","body_acc_z","body_gyro_x","body_gyro_y","body_gyro_z","total_acc_x","total_acc_y","total_acc_z"]
SEED = 42; N_CLASSES = 6; EPOCHS = 30; BATCH = 32; DROPOUT = 0.2; DENSE = 16; LR = 1e-3
CACHE = '/tmp/w/ex/har_data.npz'

def load_split(split):
    ch = [np.loadtxt(f'{U}/{split}/Inertial Signals/{s}_{split}.txt', dtype=np.float32) for s in SIGNALS]
    X = np.stack(ch, axis=-1); y = np.loadtxt(f'{U}/{split}/y_{split}.txt', dtype=np.int64) - 1
    return X, y

def stratified_subsample(X, y, target_n, seed=SEED):
    rng = np.random.RandomState(seed); per = target_n // N_CLASSES; idx = []
    for c in range(N_CLASSES):
        ic = np.where(y == c)[0]; rng.shuffle(ic); idx.append(ic[:min(per, len(ic))])
    idx = np.concatenate(idx); rng.shuffle(idx)
    return X[idx], y[idx]

def get_data():
    if os.path.exists(CACHE):
        d = np.load(CACHE); return {k: d[k] for k in d.files}
    Xtr_full, ytr_full = load_split('train'); Xte, yte = load_split('test')
    Xp, yp = stratified_subsample(Xtr_full, ytr_full, 2400)
    Xtr, Xva, ytr, yva = train_test_split(Xp, yp, test_size=15/85, stratify=yp, random_state=SEED)
    sc = StandardScaler().fit(Xtr.reshape(-1, 9))
    f = lambda X: sc.transform(X.reshape(-1, 9)).reshape(X.shape).astype(np.float32)
    d = dict(Xtr=f(Xtr), Xva=f(Xva), Xte=f(Xte), ytr=ytr, yva=yva, yte=yte)
    np.savez(CACHE, **d); return d

def build(cell, units=32, n_layers=1, bidir=False, T=128, name='m'):
    cls = dict(rnn=layers.SimpleRNN, lstm=layers.LSTM, gru=layers.GRU)[cell]
    inp = layers.Input(shape=(T, 9), name='sequence_input'); x = inp
    for k in range(n_layers):
        rs = k < n_layers - 1
        layer = cls(units, return_sequences=rs, name=f'recurrent_{k}')
        x = layers.Bidirectional(layer)(x) if bidir else layer(x)
    x = layers.Dropout(DROPOUT)(x); x = layers.Dense(DENSE, activation='relu')(x)
    out = layers.Dense(N_CLASSES, activation='softmax', dtype='float32')(x)
    m = keras.Model(inp, out, name=name)
    m.compile(optimizer=optimizers.Adam(learning_rate=LR), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

def run(name, cell, seed, out_dir='/tmp/w/ex/results', **kw):
    path = f'{out_dir}/{name}_s{seed}.json'
    if os.path.exists(path): return json.load(open(path))
    d = get_data(); T = kw.get('T', 128)
    tf.keras.utils.set_random_seed(seed)
    m = build(cell, name=name, **kw)
    t0 = time.time()
    h = m.fit(d['Xtr'][:, :T], d['ytr'], validation_data=(d['Xva'][:, :T], d['yva']), epochs=EPOCHS, batch_size=BATCH, verbose=0)
    dt = time.time() - t0
    yp = m.predict(d['Xte'][:, :T], batch_size=256, verbose=0).argmax(1); yt = d['yte']
    r = dict(name=name, cell=cell, seed=seed, kw=kw, params=int(sum(np.prod(w.shape) for w in m.trainable_weights)),
             train_time_s=dt, acc=accuracy_score(yt, yp)*100,
             prec=precision_score(yt, yp, average='macro', zero_division=0)*100,
             rec=recall_score(yt, yp, average='macro', zero_division=0)*100,
             f1=f1_score(yt, yp, average='macro', zero_division=0)*100,
             hist={k: [float(v) for v in vs] for k, vs in h.history.items()},
             cm=confusion_matrix(yt, yp, labels=range(6)).tolist())
    json.dump(r, open(path, 'w'))
    return r
