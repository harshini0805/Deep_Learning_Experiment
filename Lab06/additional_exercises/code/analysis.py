import json, glob, os, statistics as st
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib as mpl
R = '/tmp/w/ex/results'; F = '/tmp/w/ex/fig'
COL = {'rnn': '#2a78d6', 'lstm': '#eb6834', 'gru': '#1baf7a'}          # slots 1-3 of the validated palette, fixed per entity
MK = {'rnn': 'o', 'lstm': 's', 'gru': '^'}; LAB = {'rnn': 'RNN', 'lstm': 'LSTM', 'gru': 'GRU'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'DejaVu Serif'], 'font.weight': 'bold',
    'axes.titleweight': 'bold', 'axes.labelweight': 'bold', 'axes.titlesize': 13, 'axes.labelsize': 12, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'legend.fontsize': 10, 'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.edgecolor': INK2,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
    'figure.dpi': 100, 'savefig.dpi': 200, 'savefig.bbox': 'tight', 'lines.linewidth': 2, 'lines.markersize': 8})
def load(prefix):
    out = {}
    for f in sorted(glob.glob(f'{R}/{prefix}_*_s[0-9]*.json')):
        r = json.load(open(f)); out.setdefault(r['name'], []).append(r)
    return out
def ms(v):
    v = list(v); return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
def agg(rs):
    return {k: ms([r[k] for r in rs]) for k in ['acc', 'prec', 'rec', 'f1', 'train_time_s']} | {'params': rs[0]['params'], 'n': len(rs), 'per_seed_f1': [r['f1'] for r in rs], 'per_seed_acc': [r['acc'] for r in rs], 'per_seed_time': [r['train_time_s'] for r in rs]}
S = {}
ex1 = load('ex1'); ex3 = load('ex3'); ex4 = load('ex4')
S['ex1'] = {n: agg(v) for n, v in ex1.items()}; S['ex3'] = {n: agg(v) for n, v in ex3.items()}; S['ex4'] = {n: agg(v) for n, v in ex4.items()}
# ---- fig ex1: F1 and time vs units ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
for c in ['rnn', 'lstm', 'gru']:
    us = [16, 32, 64]
    for k, (a, key) in enumerate(zip(ax, ['f1', 'train_time_s'])):
        m = [S['ex1'][f'ex1_{c}_u{u}'][key][0] for u in us if f'ex1_{c}_u{u}' in S['ex1']]
        e = [S['ex1'][f'ex1_{c}_u{u}'][key][1] for u in us if f'ex1_{c}_u{u}' in S['ex1']]
        xs = [u for u in us if f'ex1_{c}_u{u}' in S['ex1']]
        a.errorbar(xs, m, yerr=e, color=COL[c], marker=MK[c], capsize=4, label=LAB[c], markeredgecolor='white', markeredgewidth=1.5)
        if xs and k == 1: a.annotate(LAB[c], (xs[-1], m[-1]), textcoords='offset points', xytext=(8, 0), va='center', color=INK, fontsize=10)
for a, t, yl in zip(ax, ['Test macro F1 vs. recurrent units', 'Training time vs. recurrent units'], ['Test macro F1 (%)', 'Training time, 30 epochs (s)']):
    a.set_title(t); a.set_xlabel('Recurrent units'); a.set_ylabel(yl); a.set_xscale('log', base=2); a.set_xticks([16, 32, 64]); a.set_xticklabels(['16', '32', '64']); a.set_xlim(13, 85)
ax[0].legend(loc='center left', bbox_to_anchor=(0.0, 0.45), frameon=False)
plt.tight_layout(); plt.savefig(f'{F}/ex1_units.png'); plt.close()
# ---- fig ex3/4: depth and bidirectionality ----
bars = [('RNN', 'rnn', S['ex1'].get('ex1_rnn_u32'), ''), ('RNN, 2 layers', 'rnn', S['ex3'].get('ex3_rnn_L2'), '//'),
        ('LSTM', 'lstm', S['ex1'].get('ex1_lstm_u32'), ''), ('LSTM, 2 layers', 'lstm', S['ex3'].get('ex3_lstm_L2'), '//'), ('BiLSTM', 'lstm', S['ex4'].get('ex4_bilstm_u32'), 'xx'),
        ('GRU', 'gru', S['ex1'].get('ex1_gru_u32'), ''), ('GRU, 2 layers', 'gru', S['ex3'].get('ex3_gru_L2'), '//')]
bars = [b for b in bars if b[2]]
fig, a = plt.subplots(figsize=(7.2, 4.4))
MKV = {'': 'o', '//': 'D', 'xx': 's'}
for i, (lab, c, st, h) in enumerate(bars):
    a.errorbar(i, st['f1'][0], yerr=st['f1'][1], color=COL[c], marker=MKV[h], markersize=10, capsize=5, lw=2, markeredgecolor='white', markeredgewidth=1.5, linestyle='none')
    a.annotate(f"{st['f1'][0]:.1f}", (i, st['f1'][0]), textcoords='offset points', xytext=(12, -3), fontsize=10, color=INK)
a.set_xticks(range(len(bars))); a.set_xticklabels([b[0] for b in bars], rotation=30, ha='right'); a.set_xlim(-0.6, len(bars) - 0.4); a.set_ylabel('Test macro F1 (%)'); a.set_ylim(45, 100)
a.set_title('Effect of depth and bidirectionality (32 units; mean \u00b1 s.d., n=3)'); a.grid(axis='x', visible=False)
plt.tight_layout(); plt.savefig(f'{F}/ex3_ex4_depth.png'); plt.close()
# ---- ex5: cost vs T ----
user_t = {'rnn': [21.0, 29.0, 45.8], 'lstm': [32.8, 53.7, 93.4], 'gru': [36.8, 66.2, 126.7]}
S['ex5_user'] = user_t
if os.path.exists(f'{R}/bench.json'):
    B = json.load(open(f'{R}/bench.json')); S['ex5_bench'] = B
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3)); Ts = [32, 64, 128]
    for c in ['rnn', 'lstm', 'gru']:
        ax[0].plot(Ts, user_t[c], color=COL[c], marker=MK[c], label=LAB[c], markeredgecolor='white', markeredgewidth=1.5)
        ax[1].plot(Ts, [B[f'{c}_T{T}']['epoch_s_min'] for T in Ts], color=COL[c], marker=MK[c], label=LAB[c], markeredgecolor='white', markeredgewidth=1.5)
        for a_, ys in [(ax[0], user_t[c]), (ax[1], [B[f'{c}_T{T}']['epoch_s_min'] for T in Ts])]:
            a_.annotate(LAB[c], (128, ys[-1]), textcoords='offset points', xytext=(8, 0), va='center', fontsize=10)
    for a_, t, yl in zip(ax, ['Recorded 30-epoch training time', 'Controlled benchmark: best-of-15 epoch time'], ['Training time (s)', 'Seconds per epoch']):
        a_.set_title(t); a_.set_xlabel('Sequence length T'); a_.set_ylabel(yl); a_.set_xticks(Ts); a_.set_xlim(26, 152)
    ax[0].legend(loc='upper left', frameon=False)
    plt.tight_layout(); plt.savefig(f'{F}/ex5_cost.png'); plt.close()
# ---- ex6: video paired ----
if os.path.exists(f'{R}/video_cv.json'):
    V = json.load(open(f'{R}/video_cv.json')); S['ex6'] = V
    la = [v['lstm']['acc'] for v in V]; ga = [v['gru']['acc'] for v in V]
    fig, a = plt.subplots(figsize=(5.6, 4.4))
    for x, vals, c in [(0, la, 'lstm'), (1, ga, 'gru')]:
        cnt = {}
        for v_ in vals:
            k_ = cnt.get(v_, 0); cnt[v_] = k_ + 1
            n_same = vals.count(v_); off = (k_ - (n_same - 1) / 2) * 0.075
            a.scatter([x + off], [v_], s=80, color=COL[c], edgecolor='white', linewidth=1.2, marker=MK[c], zorder=3)
        a.hlines(np.mean(vals), x - 0.3, x + 0.3, color=INK, lw=2.5, zorder=4)
        a.annotate(f'mean {np.mean(vals):.1f}', (x + 0.3, np.mean(vals)), textcoords='offset points', xytext=(6, 0), va='center', fontsize=10)
    a.set_xticks([0, 1]); a.set_xticklabels(['CNN\u2013LSTM', 'CNN\u2013GRU']); a.set_xlim(-0.55, 1.75); a.set_ylabel('Test accuracy (%)'); a.set_title('10 stratified splits, identical features'); a.grid(axis='x', visible=False)
    plt.tight_layout(); plt.savefig(f'{F}/ex6_video.png'); plt.close()
# ---- ex7: seq2seq ----
s2 = {}
for f in sorted(glob.glob(f'{R}/s2s_*_s*.json')):
    r = json.load(open(f)); s2.setdefault(r['task'], []).append(r)
if s2:
    S['ex7'] = {t: {k: ms([r[k] for r in rs]) for k in ['token_acc', 'seq_acc', 'len_acc', 'train_loss', 'val_loss', 'train_time_s']} | {'params': rs[0]['params'], 'n': len(rs), 'examples': rs[0]['examples']} for t, rs in s2.items()}
    order = [t for t in ['reverse_6to6', 'reverse_6to3', 'mirror_4to8'] if t in s2]; names = {'reverse_6to6': 'Reverse\n6 → 6', 'reverse_6to3': 'Reverse first 3\n6 → 3', 'mirror_4to8': 'Mirror\n4 → 8'}
    fig, a = plt.subplots(figsize=(7.2, 4.4)); w = 0.26
    cols = {'token_acc': '#2a78d6', 'seq_acc': '#eb6834', 'len_acc': '#1baf7a'}; lab = {'token_acc': 'Token accuracy', 'seq_acc': 'Sequence accuracy', 'len_acc': 'Length accuracy'}
    for j, k in enumerate(['token_acc', 'seq_acc', 'len_acc']):
        m = [S['ex7'][t][k][0] for t in order]; e = [S['ex7'][t][k][1] for t in order]
        a.bar(np.arange(len(order)) + (j - 1) * (w + 0.02), m, w, yerr=e, color=cols[k], edgecolor='white', linewidth=1.2, capsize=3, label=lab[k], error_kw=dict(ecolor=INK, lw=1.3))
        for x, mm, ee in zip(np.arange(len(order)) + (j - 1) * (w + 0.02), m, e): a.text(x, mm + ee + 1.2, f'{mm:.1f}', ha='center', fontsize=8.5)
    a.set_xticks(range(len(order))); a.set_xticklabels([names[t] for t in order]); a.set_ylim(0, 108); a.set_yticks([0, 20, 40, 60, 80, 100]); a.set_ylabel('Test accuracy (%)'); a.grid(axis='x', visible=False)
    a.set_title('Sequence-to-sequence: output length vs. input length'); a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False, ncol=3)
    plt.tight_layout(); plt.savefig(f'{F}/ex7_s2s.png'); plt.close()
json.dump(S, open('/tmp/w/ex/summary.json', 'w'), indent=1)
print('analysis ok; keys', list(S.keys()))
