"""Generate Polish report directly from recorded metrics; never invent results."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
with (R/'results/metrics.csv').open(encoding='utf-8', newline='') as f: rows=list(csv.DictReader(f))
test=[r for r in rows if r['split']=='test']
assert len(test)==9
modes=['baseline','oversampling','augmentation']
labels={'baseline':'Bazowy','oversampling':'Oversampling','augmentation':'Augmentacja'}
metrics=['mau_recall','macro_f1','accuracy','mau_precision','mau_f1']

def vals(mode,key):return np.array([float(r[key]) for r in test if r['condition']==mode])
def pct(v):return f'{100*v:.2f}'.replace('.',',')
text=['# Wyniki: augmentacja Egyptian Mau','',
'Raport z rzeczywiście wykonanych 9 przebiegów (3 metody × 3 losowe podziały). Zamrożony ResNet-18 ImageNet + regresja logistyczna. Data wykonania: 18.10.2026. Pełne szczegóły i ograniczenia w `protocol.md`.','',
'## Wyniki testowe każdego przebiegu','',
'Wszystkie wartości w tabelach poniżej podano w procentach. Każdy test obejmuje 240 obrazów, w tym 30 Egyptian Mau.','',
'| Iteracja | Metoda | Recall Mau | Macro-F1 | Accuracy | Precision Mau | F1 Mau |',
'|---|---|---:|---:|---:|---:|---:|']
for r in test:text.append('| '+r['iteration']+' | '+labels[r['condition']]+' | '+' | '.join(pct(float(r[k])) for k in metrics)+' |')
text += ['', '## Średnia ± odchylenie standardowe', '', 'Odchylenie standardowe próby (ddof=1), n=3; nie jest to przedział ufności.','',
'| Metoda | Recall Mau | Macro-F1 | Accuracy |','|---|---:|---:|---:|']
aggregate=[]
for m in modes:
    text.append('| '+labels[m]+' | '+' | '.join(f'{pct(vals(m,k).mean())} ± {pct(vals(m,k).std(ddof=1))}' for k in metrics[:3])+' |')
    for k in metrics:aggregate.append(dict(condition=m,metric=k,mean=vals(m,k).mean(),sample_std=vals(m,k).std(ddof=1),n=3))
with (R/'results/summary.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=aggregate[0]);w.writeheader();w.writerows(aggregate)
text += ['', '## Porównania sparowane', '', 'Różnice w punktach procentowych, liczone wewnątrz każdej iteracji. Dodatnia wartość oznacza poprawę.','',
'| Porównanie | Δ recall: iteracje 1 / 2 / 3 | Średnia Δ recall | Średnia Δ macro-F1 |','|---|---:|---:|---:|']
for a,b in [('oversampling','baseline'),('augmentation','baseline'),('augmentation','oversampling')]:
    dr=vals(a,'mau_recall')-vals(b,'mau_recall');df=vals(a,'macro_f1')-vals(b,'macro_f1')
    text.append(f'| {labels[a]} − {labels[b]} | '+ ' / '.join(pct(v) for v in dr)+f' | {pct(dr.mean())} | {pct(df.mean())} |')
dr=vals('augmentation','mau_recall')-vals('baseline','mau_recall')
do=vals('augmentation','mau_recall')-vals('oversampling','mau_recall')
df=vals('augmentation','macro_f1')-vals('baseline','macro_f1')
text += ['', '## Interpretacja','',
f'Względem wariantu bazowego augmentacja zmieniła średni recall Egyptian Mau o {pct(dr.mean())} p.p. i macro-F1 o {pct(df.mean())} p.p. Recall wzrósł w {int((dr>0).sum())} z 3 iteracji, pozostał bez zmian w {int((dr==0).sum())}, a spadł w {int((dr<0).sum())}.', '',
f'W porównaniu z samym powielaniem zdjęć różnica średniego recall wyniosła {pct(do.mean())} p.p. na rzecz augmentacji (wartość ujemna oznacza przewagę oversamplingu). To porównanie rozdziela efekt większej liczby obserwacji od efektu zmiany ich wyglądu. Budżet i źródła dodatkowych obserwacji są identyczne.','']
if dr.mean()>0:
    text.append('Eksperyment pokazuje, że w tym ustawieniu augmentacja może poprawić średnią wykrywalność niedoreprezentowanej klasy. Nie oznacza to gwarancji poprawy przy innym podziale, transformacjach lub klasyfikatorze.')
else:
    text.append('W tym ustawieniu nie uzyskano poprawy średniego recall mniejszości dzięki augmentacji. Nie dowodzi to, że augmentacja nigdy nie pomaga: wynik dotyczy wybranych transformacji, reprezentacji, klasyfikatora i puli danych.')
text += ['', 'Spełnienie celu wymaga również kontroli jakości globalnej. '+('Średnie macro-F1 nie spadło względem bazowego.' if df.mean()>=0 else 'Średnie macro-F1 spadło względem bazowego, więc ewentualna poprawa mniejszości ma koszt globalny.'), '',
'## Walidacja i ograniczenia','',
'Walidacja została oceniona dla każdego modelu i zapisana osobno. Nie służyła do zmieniania C ani do dobierania transformacji: parametry ustalono z góry. Testu nie wykorzystano do strojenia.','',
'Jedna poprawnie sklasyfikowana fotografia Egyptian Mau to 3,33 p.p. recall. Przy 30 przykładach tej klasy i trzech nakładających się losowych podziałach oszacowania są niestabilne. Nie przedstawiamy testu istotności ani mocnych wniosków populacyjnych. Odchylenie między iteracjami opisuje wrażliwość na podział, a nie niezależne powtórzenia na nowych populacjach.','',
'Kontrole wykazały rozłączność oryginałów i rodziców augmentacji wewnątrz każdego podziału. Nie eliminowano podobnych, ale nieidentycznych zdjęć ani tożsamości tych samych zwierząt. Użyto tylko jednej architektury i klasycznej augmentacji; nie wykonano eksperymentu dyfuzyjnego ani porównania z EfficientNet-B0.','',
'## Pliki dowodowe','',
'`results/metrics.csv` zawiera wszystkie wyniki testowe i walidacyjne, `results/summary.csv` agregaty. Katalogi każdego przebiegu zawierają pełne raporty ośmiu klas, macierze pomyłek, predykcje z prawdopodobieństwami, listę obserwacji treningowych i parametry klasyfikatora. `results/leakage_check.json` oraz `class_counts.csv` dokumentują kontrole.','',
'![Wyniki testowe](results/comparison.png)','', 'Przykłady: oryginały u góry, augmentacje poniżej.', '', '![Przykłady augmentacji](results/augmentation_examples.jpg)', '',
'Źródła, licencja i protokół: `protocol.md` oraz `DATA_LICENSE.md`.']
(R/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for ax,k,title in zip(axes,['mau_recall','macro_f1'],['Egyptian Mau recall','Macro-F1 (8 classes)']):
    for i in range(3):ax.plot(range(3),[vals(m,k)[i]*100 for m in modes],marker='o',label=f'Iteration {i+1}',alpha=.75)
    ax.set(xticks=range(3),xticklabels=[labels[m] for m in modes],ylabel='%',title=title,ylim=(0,100));ax.grid(alpha=.2);ax.legend()
fig.tight_layout();fig.savefig(R/'results/comparison.png',dpi=160)
print('Report written to report.md')
