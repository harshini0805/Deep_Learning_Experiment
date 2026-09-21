import sys, json, time, statistics; sys.path.insert(0, '/tmp/w/ex')
import lab, numpy as np, tensorflow as tf
d = lab.get_data(); models = {}
for cell in ['rnn', 'lstm', 'gru']:
    for T in [32, 64, 128]:
        tf.keras.utils.set_random_seed(42); m = lab.build(cell, T=T); m.fit(d['Xtr'][:, :T], d['ytr'], epochs=2, batch_size=32, verbose=0); models[(cell, T)] = m  # 2 warm-up epochs
ts = {k: [] for k in models}
for rep in range(15):                       # interleave configs so machine-wide slowdowns hit every config equally
    for (cell, T), m in models.items():
        t0 = time.time(); m.fit(d['Xtr'][:, :T], d['ytr'], epochs=1, batch_size=32, verbose=0); ts[(cell, T)].append(time.time() - t0)
out = {}
for (cell, T), v in ts.items():
    m = models[(cell, T)]
    out[f'{cell}_T{T}'] = dict(cell=cell, T=T, params=int(sum(np.prod(w.shape) for w in m.trainable_weights)), epoch_s_min=min(v), epoch_s_median=statistics.median(v), epoch_s_all=v)
    print('BENCH', cell, T, 'min', round(min(v), 3), 'median', round(statistics.median(v), 3), flush=True)
json.dump(out, open('/tmp/w/ex/results/bench.json', 'w')); print('ALL_BENCH_DONE', flush=True)
