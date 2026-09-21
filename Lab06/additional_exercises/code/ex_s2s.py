import os, sys, json, time; sys.path.insert(0, '/tmp/w/ex')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from sklearn.model_selection import train_test_split
MAXV = 20; PAD, SOS, EOS = 0, MAXV + 1, MAXV + 2; V = MAXV + 3; EMB = 32; UNITS = 64; N = 8000; SEED = 42
TASKS = {  # name: (input_len, target_fn)
    'reverse_6to6':  (6, lambda s: s[:, ::-1]),                                  # control (notebook task)
    'reverse_6to3':  (6, lambda s: s[:, ::-1][:, :3]),                           # shorter output (3 < 6)
    'mirror_4to8':   (4, lambda s: np.concatenate([s, s[:, ::-1]], axis=1)),     # longer output (8 > 4)
}
def make(task):
    L, fn = TASKS[task]; rng = np.random.RandomState(SEED)
    src = rng.randint(1, MAXV + 1, size=(N, L)); tgt = fn(src).copy(); n = len(src)
    din = np.concatenate([np.full((n, 1), SOS), tgt], 1); dtg = np.concatenate([tgt, np.full((n, 1), EOS)], 1)
    a = [src.astype('int32'), din.astype('int32'), dtg.astype('int32'), tgt]
    tr = train_test_split(*a, test_size=0.2, random_state=SEED)
    # tr = [src_tr, src_te, din_tr, din_te, dtg_tr, dtg_te, tgt_tr, tgt_te]
    s_tr, s_te, di_tr, di_te, dt_tr, dt_te, g_tr, g_te = tr
    s_tr, s_va, di_tr, di_va, dt_tr, dt_va, g_tr, g_va = train_test_split(s_tr, di_tr, dt_tr, g_tr, test_size=0.125, random_state=SEED)
    return dict(s_tr=s_tr, s_va=s_va, s_te=s_te, di_tr=di_tr, di_va=di_va, dt_tr=dt_tr, dt_va=dt_va, g_te=g_te)
def build():
    ei = layers.Input(shape=(None,)); ee = layers.Embedding(V, EMB, mask_zero=True)
    enc = layers.LSTM(UNITS, return_state=True); _, sh, sc = enc(ee(ei))
    di = layers.Input(shape=(None,)); de = layers.Embedding(V, EMB, mask_zero=True)
    dec = layers.LSTM(UNITS, return_sequences=True, return_state=True); do, _, _ = dec(de(di), initial_state=[sh, sc])
    dn = layers.Dense(V, activation='softmax', dtype='float32'); out = dn(do)
    tm = keras.Model([ei, di], out); tm.compile(optimizer=optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    enc_m = keras.Model(ei, [sh, sc])
    ih, ic, it = layers.Input(shape=(UNITS,)), layers.Input(shape=(UNITS,)), layers.Input(shape=(1,))
    o, h, c = dec(de(it), initial_state=[ih, ic]); dec_m = keras.Model([it, ih, ic], [dn(o), h, c])
    return tm, enc_m, dec_m
def decode(enc_m, dec_m, src, max_len):
    h, c = enc_m.predict(src, batch_size=512, verbose=0); B = len(src)
    tok = np.full((B, 1), SOS); done = np.zeros(B, bool); outs = [[] for _ in range(B)]
    for _ in range(max_len):
        p, h, c = dec_m.predict([tok, h, c], batch_size=512, verbose=0); nt = p[:, -1].argmax(-1)
        for i in range(B):
            if not done[i]:
                if nt[i] == EOS: done[i] = True
                else: outs[i].append(int(nt[i]))
        tok = nt[:, None]
        if done.all(): break
    return outs
def run(task, seed):
    path = f'/tmp/w/ex/results/s2s_{task}_s{seed}.json'
    if os.path.exists(path): return json.load(open(path))
    d = make(task); tf.keras.utils.set_random_seed(seed)
    tm, enc_m, dec_m = build(); t0 = time.time()
    h = tm.fit([d['s_tr'], d['di_tr']], d['dt_tr'], validation_data=([d['s_va'], d['di_va']], d['dt_va']), epochs=25, batch_size=64, verbose=0)
    dt = time.time() - t0; tgt = d['g_te']; L_out = tgt.shape[1]
    pred = decode(enc_m, dec_m, d['s_te'], L_out + 1)
    tok_c = tok_t = seq_c = len_c = 0; ex = []
    for i, p in enumerate(pred):
        t = list(tgt[i]); k = min(len(t), len(p))
        tok_c += sum(int(a == b) for a, b in zip(t[:k], p[:k])); tok_t += len(t); seq_c += int(p == t); len_c += int(len(p) == len(t))
        if i < 5: ex.append(dict(inp=[int(x) for x in d['s_te'][i]], exp=[int(x) for x in t], pred=p))
    n = len(pred)
    r = dict(task=task, seed=seed, params=int(tm.count_params()), train_time_s=dt, token_acc=tok_c / tok_t * 100, seq_acc=seq_c / n * 100, len_acc=len_c / n * 100,
             n_eval=n, train_loss=float(h.history['loss'][-1]), val_loss=float(h.history['val_loss'][-1]), hist={k: [float(v) for v in vs] for k, vs in h.history.items()}, examples=ex)
    json.dump(r, open(path, 'w')); return r
if __name__ == '__main__':
    t0 = time.time()
    for seed in [42, 43, 44]:
        for task in TASKS:
            r = run(task, seed)
            print(f"DONE s2s {task} seed={seed} params={r['params']} tok={r['token_acc']:.2f} seq={r['seq_acc']:.2f} len={r['len_acc']:.2f} t={r['train_time_s']:.0f}s elapsed={(time.time()-t0)/60:.1f}min", flush=True)
    print('ALL_S2S_DONE', flush=True)
