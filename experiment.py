"""Oxford-IIIT Pet: reproducible paired imbalance experiment. See protocol.md."""
from pathlib import Path
import argparse, csv, hashlib, io, json, platform, tarfile, urllib.request
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

ROOT = Path(__file__).resolve().parent
CLASSES = ['Abyssinian','Bengal','Birman','Bombay','Egyptian_Mau','Ragdoll','Russian_Blue','Siamese']
SEEDS = [1729, 2718, 31415]
MINORITY = 4
URL = 'https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz'

def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')

def csvwrite(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def sha(b): return hashlib.sha256(b).hexdigest()

def prepare(archive):
    if not archive.exists(): urllib.request.urlretrieve(URL, archive)
    pools={c:[] for c in CLASSES}
    with tarfile.open(archive, mode='r|gz') as tar:
        for m in tar:
            name=Path(m.name).name
            if m.isfile() and name.endswith('.jpg') and name.rsplit('_',1)[0] in pools:
                pools[name.rsplit('_',1)[0]].append((name, tar.extractfile(m).read()))
        rng=np.random.default_rng(20261018); rows=[]; seen=set()
        for label,c in enumerate(CLASSES):
            candidates=sorted(pools[c],key=lambda x:x[0])
            count=0
            for idx in rng.permutation(len(candidates)):
                name,raw=candidates[idx]
                try:
                    im=Image.open(io.BytesIO(raw)).convert('RGB'); im.load()
                except Exception: continue
                pixhash=sha(str(im.size).encode()+im.tobytes())
                if pixhash in seen: continue
                seen.add(pixhash)
                path=ROOT/'data'/'selected'/name
                path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(raw)
                rows.append(dict(path=path.relative_to(ROOT).as_posix(),label=label,breed=c,sha256=sha(raw),pixel_sha256=pixhash))
                count+=1
                if count==(70 if label==MINORITY else 120): break
            assert count==(70 if label==MINORITY else 120), c
    csvwrite(ROOT/'data'/'selection.csv',rows)
    dump(ROOT/'data'/'selection_info.json',dict(seed=20261018,source=URL,archive_sha256=sha(archive.read_bytes()),n=len(rows)))
    print('Selected',len(rows),flush=True)

def read_selection():
    with (ROOT/'data'/'selection.csv').open(encoding='utf-8', newline='') as f: rows=list(csv.DictReader(f))
    for r in rows:r['label']=int(r['label'])
    return rows

def make_splits(rows):
    manifests=[]
    for iteration,seed in enumerate(SEEDS,1):
        rng=np.random.default_rng(seed); out=[]
        for label in range(8):
            pool=[r for r in rows if r['label']==label]
            for j,idx in enumerate(rng.permutation(len(pool))):
                out.append(dict(**pool[idx],split='test' if j<30 else 'val' if j<50 else 'train'))
        csvwrite(ROOT/'manifests'/f'iteration_{iteration}.csv',out)
        manifests.append(out)
    return manifests

def added_images(rows, iteration):
    # Identical source schedule for oversampling and augmentation; every source occurs 2 or 3 times.
    rng=np.random.default_rng(SEEDS[iteration-1]+100000)
    minority=[r for r in rows if r['split']=='train' and r['label']==MINORITY]
    schedule=list(rng.permutation(20))+list(rng.permutation(20))+list(rng.permutation(20)[:10])
    added=[]
    for k,idx in enumerate(schedule):
        src=minority[idx]; im=Image.open(ROOT/src['path']).convert('RGB')
        # Four ordinary transformations: crop/zoom, rotation, mirror, brightness/contrast.
        w,h=im.size; scale=float(rng.uniform(.85,.98)); cw,ch=int(w*scale),int(h*scale)
        left=int(rng.integers(0,w-cw+1)); top=int(rng.integers(0,h-ch+1))
        angle=float(rng.uniform(-15,15)); bright=float(rng.uniform(.85,1.15)); contrast=float(rng.uniform(.85,1.15))
        mirror=bool(k%2==0)
        im=im.crop((left,top,left+cw,top+ch)).resize((256,256),Image.Resampling.BICUBIC)
        # Reflect padding avoids black corners. Center crop after rotation removes most edge effects.
        a=np.pad(np.asarray(im),((40,40),(40,40),(0,0)),mode='reflect')
        im=Image.fromarray(a).rotate(angle,Image.Resampling.BICUBIC).crop((40,40,296,296))
        if mirror: im=ImageOps.mirror(im)
        im=ImageEnhance.Brightness(im).enhance(bright)
        im=ImageEnhance.Contrast(im).enhance(contrast)
        path=ROOT/'data'/'augmented'/f'iteration_{iteration}'/f'aug_{k:02d}.png'
        path.parent.mkdir(parents=True,exist_ok=True);im.save(path)
        added.append(dict(path=path.relative_to(ROOT).as_posix(),source=src['path'],label=MINORITY,
                          crop=[left,top,cw,ch],angle=angle,mirror=mirror,brightness=bright,contrast=contrast))
    dump(ROOT/'manifests'/f'added_{iteration}.json',added)
    return added

def check(rows, manifests, added):
    assert len(rows)==910
    assert len({r['pixel_sha256'] for r in rows})==910
    for r in rows: assert sha((ROOT/r['path']).read_bytes())==r['sha256']
    counts=[]
    for i,(split,extra) in enumerate(zip(manifests,added),1):
        assert len(split)==910 and len({r['path'] for r in split})==910
        groups={s:{r['path'] for r in split if r['split']==s} for s in ['train','val','test']}
        assert not (groups['train']&groups['val'] or groups['train']&groups['test'] or groups['val']&groups['test'])
        assert len(extra)==50 and all(r['source'] in groups['train'] for r in extra)
        for label,c in enumerate(CLASSES):
            n={s:sum(r['split']==s and r['label']==label for r in split) for s in groups}
            assert n==dict(train=20 if label==MINORITY else 70,val=20,test=30)
            counts.append(dict(iteration=i,breed=c,**n,oversampling_train=70,augmentation_train=70))
    csvwrite(ROOT/'results'/'class_counts.csv',counts)
    dump(ROOT/'results'/'leakage_check.json',dict(passed=True,unique_real_images=910,iterations=3,
        note='Checks IDs, original file hashes, decoded pixel hashes and augmentation parents; near duplicates or repeated animal identities are not ruled out.'))

def extract(paths, threads):
    import torch, torchvision
    from torchvision.models import resnet18, ResNet18_Weights
    torch.set_num_threads(threads);torch.manual_seed(42);torch.use_deterministic_algorithms(True)
    torch.hub.set_dir(str(ROOT/'cache'/'torch'))
    weights=ResNet18_Weights.IMAGENET1K_V1
    model=resnet18(weights=weights);model.fc=torch.nn.Identity();model.eval()
    transform=weights.transforms(); features=[]
    with torch.inference_mode():
        for start in range(0,len(paths),32):
            batch=torch.stack([transform(Image.open(ROOT/p).convert('RGB')) for p in paths[start:start+32]])
            features.append(model(batch).numpy())
            print(f'Features {min(start+32,len(paths))}/{len(paths)}',flush=True)
    return np.concatenate(features),dict(torch=torch.__version__,torchvision=torchvision.__version__,weights=str(weights),weights_sha256=sha((ROOT/'cache/torch/checkpoints/resnet18-f37072fd.pth').read_bytes()),threads=threads)

def evaluate(manifests, added, paths, X):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
    import sklearn, warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings('error',category=ConvergenceWarning)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lookup={p:i for i,p in enumerate(paths)}; summaries=[]
    for i,(rows,extra) in enumerate(zip(manifests,added),1):
        train=[r for r in rows if r['split']=='train']
        # Fit normalization ONCE on the original train split; identical across conditions.
        scaler=StandardScaler().fit(X[[lookup[r['path']] for r in train]])
        for mode in ['baseline','oversampling','augmentation']:
            current=train.copy()
            if mode!='baseline': current += [dict(path=r['source'] if mode=='oversampling' else r['path'],label=MINORITY) for r in extra]
            out=ROOT/'results'/f'iteration_{i}'/mode;out.mkdir(parents=True,exist_ok=True)
            csvwrite(out/'train_observations.csv',[dict(observation=j,path=r['path'],label=r['label']) for j,r in enumerate(current)])
            model=LogisticRegression(C=1.0,solver='lbfgs',max_iter=3000,tol=1e-7)
            model.fit(scaler.transform(X[[lookup[r['path']] for r in current]]),[r['label'] for r in current])
            np.savez_compressed(out/'classifier.npz',coef=model.coef_,intercept=model.intercept_,classes=model.classes_,mean=scaler.mean_,scale=scaler.scale_)
            for split in ['val','test']:
                sample=[r for r in rows if r['split']==split]; y=[r['label'] for r in sample]
                proba=model.predict_proba(scaler.transform(X[[lookup[r['path']] for r in sample]]));pred=proba.argmax(1)
                report=classification_report(y,pred,labels=list(range(8)),target_names=CLASSES,output_dict=True,zero_division=0)
                dump(out/f'{split}_metrics.json',report)
                csvwrite(out/f'{split}_predictions.csv',[dict(path=r['path'],true=r['label'],predicted=int(p),**{f'p_{c}':float(v) for c,v in zip(CLASSES,prob)}) for r,p,prob in zip(sample,pred,proba)])
                cm=confusion_matrix(y,pred,labels=list(range(8)))
                csvwrite(out/f'{split}_confusion.csv',[dict(actual=c,**dict(zip(CLASSES,map(int,row)))) for c,row in zip(CLASSES,cm)])
                summaries.append(dict(iteration=i,seed=SEEDS[i-1],condition=mode,split=split,n_train=len(current),accuracy=accuracy_score(y,pred),macro_f1=f1_score(y,pred,average='macro'),mau_recall=report['Egyptian_Mau']['recall'],mau_precision=report['Egyptian_Mau']['precision'],mau_f1=report['Egyptian_Mau']['f1-score']))
                if split=='test':
                    fig,ax=plt.subplots(figsize=(8,7));ax.imshow(cm,cmap='Blues',vmin=0,vmax=30)
                    ax.set(xticks=range(8),yticks=range(8),xticklabels=CLASSES,yticklabels=CLASSES,xlabel='Predicted',ylabel='Actual',title=f'Iteration {i} — {mode}')
                    plt.setp(ax.get_xticklabels(),rotation=45,ha='right')
                    for a in range(8):
                        for b in range(8):ax.text(b,a,str(cm[a,b]),ha='center',va='center',color='white' if cm[a,b]>15 else 'black')
                    fig.tight_layout();fig.savefig(out/'test_confusion.png',dpi=130);plt.close(fig)
            print(i,mode,summaries[-1],flush=True)
    csvwrite(ROOT/'results'/'metrics.csv',summaries)
    return summaries

def main():
    p=argparse.ArgumentParser(description='Compare baseline, oversampling, and augmentation across three fixed data splits.');p.add_argument('--archive',type=Path,default=ROOT.parent/'images.tar.gz');p.add_argument('--prepare',action='store_true');p.add_argument('--rebuild-data',action='store_true');p.add_argument('--threads',type=int,default=4);p.add_argument('--recompute-features',action='store_true');args=p.parse_args()
    if args.prepare or not (ROOT/'data'/'selection.csv').exists():prepare(args.archive)
    rows=read_selection()
    missing = [r['path'] for r in rows if not (ROOT / r['path']).is_file()]
    if missing:
        p.error(f'{len(missing)} original images are missing from data/selected/. Supply the selected JPEG subset or use --prepare --recompute-features with the feature-extraction dependencies installed.')
    if args.prepare or args.rebuild_data or not (ROOT/'manifests'/'iteration_1.csv').exists():
        manifests=make_splits(rows);added=[added_images(m,i) for i,m in enumerate(manifests,1)]
    else:
        manifests=[];added=[]
        for i in range(1,4):
            with (ROOT/'manifests'/f'iteration_{i}.csv').open(encoding='utf-8',newline='') as f:
                part=list(csv.DictReader(f))
            for r in part:r['label']=int(r['label'])
            manifests.append(part)
            added.append(json.loads((ROOT/'manifests'/f'added_{i}.json').read_text(encoding='utf-8')))
    check(rows,manifests,added)
    paths=[r['path'] for r in rows]+[r['path'] for e in added for r in e]
    fingerprints=[sha((ROOT/p).read_bytes()) for p in paths]
    cache=ROOT/'features.npz';metadata=ROOT/'results'/'environment.json'
    if cache.exists() and not args.recompute_features:
        with np.load(cache) as a:
            assert list(a['paths'])==paths and list(a['sha256'])==fingerprints, 'Stale feature cache: use --recompute-features'
            X=a['X']
    else:
        X,env=extract(paths,args.threads)
        np.savez_compressed(cache,paths=paths,sha256=fingerprints,X=X)
        import importlib.metadata as md
        env.update(python=platform.python_version(),platform=platform.platform(),packages={x:md.version(x) for x in ['numpy','Pillow','scikit-learn','matplotlib']})
        dump(metadata,env)
    evaluate(manifests,added,paths,X)

if __name__=='__main__':main()
