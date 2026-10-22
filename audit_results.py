"""Independent consistency checks of saved predictions, manifests and metrics."""
from pathlib import Path
import csv,json
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,recall_score
R=Path(__file__).resolve().parent

def read(p):
    with p.open(encoding='utf-8', newline='') as f:return list(csv.DictReader(f))
metrics=read(R/'results/metrics.csv');assert len(metrics)==18
for i in range(1,4):
    rows=read(R/'manifests'/f'iteration_{i}.csv')
    train=[r for r in rows if r['split']=='train']
    original={r['path'] for r in train}
    for split,n in [('val',160),('test',240)]:
        expected={r['path']:int(r['label']) for r in rows if r['split']==split}
        for mode in ['baseline','oversampling','augmentation']:
            out=R/'results'/f'iteration_{i}'/mode
            preds=read(out/f'{split}_predictions.csv')
            assert len(preds)==n and {r['path']:int(r['true']) for r in preds}==expected
            y=np.array([int(r['true']) for r in preds]);p=np.array([int(r['predicted']) for r in preds])
            row=next(r for r in metrics if int(r['iteration'])==i and r['condition']==mode and r['split']==split)
            for k,v in [('accuracy',accuracy_score(y,p)),('macro_f1',f1_score(y,p,average='macro')),('mau_recall',recall_score(y,p,labels=[4],average=None)[0])]:
                assert np.isclose(v,float(row[k]),atol=1e-12)
            obs=read(out/'train_observations.csv')
            assert len(obs)==(510 if mode=='baseline' else 560)
            assert [r['path'] for r in obs[:510]]==[r['path'] for r in train]
            if mode=='oversampling': assert all(r['path'] in original and int(r['label'])==4 for r in obs[510:])
            if mode=='augmentation': assert len({r['path'] for r in obs})==560
result=dict(passed=True,runs=9,evaluations=18,checks=['identical validation/test IDs and labels across methods','recomputed accuracy, macro-F1 and minority recall','training counts and common 510 originals','oversampling source membership','unique augmentation observations'])
(R/'results/audit.json').write_text(json.dumps(result,indent=2)+'\n', encoding='utf-8')
print(result)
