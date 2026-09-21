import sys, time; sys.path.insert(0, '/tmp/w/ex')
import lab
SEEDS = [42, 43, 44]
jobs = []
for seed in SEEDS:
    for units in [16, 32, 64]:
        for cell in ['rnn', 'lstm', 'gru']:
            jobs.append((f'ex1_{cell}_u{units}', cell, seed, dict(units=units)))
    jobs.append(('ex4_bilstm_u32', 'lstm', seed, dict(units=32, bidir=True)))
    for cell in ['rnn', 'lstm', 'gru']:
        jobs.append((f'ex3_{cell}_L2', cell, seed, dict(units=32, n_layers=2)))
t0 = time.time()
for i, (name, cell, seed, kw) in enumerate(jobs):
    r = lab.run(name, cell, seed, **kw)
    print(f"DONE {i+1}/{len(jobs)} {name} seed={seed} params={r['params']} acc={r['acc']:.2f} f1={r['f1']:.2f} t={r['train_time_s']:.0f}s elapsed={(time.time()-t0)/60:.1f}min", flush=True)
print('ALL_HAR_DONE', flush=True)
