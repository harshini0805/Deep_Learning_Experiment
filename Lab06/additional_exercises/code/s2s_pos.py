import sys, json; sys.path.insert(0, '/tmp/w/ex')
import ex_s2s as s, numpy as np, tensorflow as tf
out = {}
for task in ['reverse_6to6', 'mirror_4to8']:
    d = s.make(task); tf.keras.utils.set_random_seed(42); tm, enc, dec = s.build()
    tm.fit([d['s_tr'], d['di_tr']], d['dt_tr'], validation_data=([d['s_va'], d['di_va']], d['dt_va']), epochs=25, batch_size=64, verbose=0)
    pred = s.decode(enc, dec, d['s_te'], d['g_te'].shape[1] + 1); tgt = d['g_te']; L = tgt.shape[1]
    pos = np.zeros(L); 
    for i, p in enumerate(pred):
        for j in range(L): pos[j] += int(j < len(p) and p[j] == tgt[i][j])
    pos = pos / len(pred) * 100
    # errors on repeated tokens: does the sequence contain duplicates?
    dup = np.array([len(set(x)) < len(x) for x in d['s_te']]); ok = np.array([list(p) == list(t) for p, t in zip(pred, tgt)])
    out[task] = dict(pos_acc=pos.tolist(), seq_acc=float(ok.mean() * 100), seq_acc_dup=float(ok[dup].mean() * 100), seq_acc_nodup=float(ok[~dup].mean() * 100), frac_dup=float(dup.mean() * 100))
    print('POS', task, [round(x, 1) for x in pos], 'seq', round(ok.mean() * 100, 1), 'dup', round(ok[dup].mean() * 100, 1), 'nodup', round(ok[~dup].mean() * 100, 1), 'fracdup', round(dup.mean() * 100, 1), flush=True)
json.dump(out, open('/tmp/w/ex/results/s2s_pos.json', 'w'))
