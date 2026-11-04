# Wyniki: augmentacja Egyptian Mau

Raport z rzeczywiście wykonanych 9 przebiegów (3 metody × 3 losowe podziały). Zamrożony ResNet-18 ImageNet + regresja logistyczna. Data wykonania: 18.10.2026. Pełne szczegóły i ograniczenia w `protocol.md`.

## Wyniki testowe każdego przebiegu

Wszystkie wartości w tabelach poniżej podano w procentach. Każdy test obejmuje 240 obrazów, w tym 30 Egyptian Mau.

| Iteracja | Metoda | Recall Mau | Macro-F1 | Accuracy | Precision Mau | F1 Mau |
|---|---|---:|---:|---:|---:|---:|
| 1 | Bazowy | 70,00 | 84,67 | 84,58 | 87,50 | 77,78 |
| 1 | Oversampling | 73,33 | 85,09 | 85,00 | 88,00 | 80,00 |
| 1 | Augmentacja | 80,00 | 85,97 | 85,83 | 85,71 | 82,76 |
| 2 | Bazowy | 60,00 | 81,86 | 81,67 | 94,74 | 73,47 |
| 2 | Oversampling | 63,33 | 82,31 | 82,08 | 95,00 | 76,00 |
| 2 | Augmentacja | 76,67 | 84,41 | 84,17 | 95,83 | 85,19 |
| 3 | Bazowy | 73,33 | 86,22 | 86,25 | 84,62 | 78,57 |
| 3 | Oversampling | 80,00 | 87,08 | 87,08 | 85,71 | 82,76 |
| 3 | Augmentacja | 80,00 | 87,48 | 87,50 | 85,71 | 82,76 |

## Średnia ± odchylenie standardowe

Odchylenie standardowe próby (ddof=1), n=3; nie jest to przedział ufności.

| Metoda | Recall Mau | Macro-F1 | Accuracy |
|---|---:|---:|---:|
| Bazowy | 67,78 ± 6,94 | 84,25 ± 2,21 | 84,17 ± 2,32 |
| Oversampling | 72,22 ± 8,39 | 84,83 ± 2,39 | 84,72 ± 2,51 |
| Augmentacja | 78,89 ± 1,92 | 85,95 ± 1,54 | 85,83 ± 1,67 |

## Porównania sparowane

Różnice w punktach procentowych, liczone wewnątrz każdej iteracji. Dodatnia wartość oznacza poprawę.

| Porównanie | Δ recall: iteracje 1 / 2 / 3 | Średnia Δ recall | Średnia Δ macro-F1 |
|---|---:|---:|---:|
| Oversampling − Bazowy | 3,33 / 3,33 / 6,67 | 4,44 | 0,58 |
| Augmentacja − Bazowy | 10,00 / 16,67 / 6,67 | 11,11 | 1,70 |
| Augmentacja − Oversampling | 6,67 / 13,33 / 0,00 | 6,67 | 1,13 |

## Interpretacja

Względem wariantu bazowego augmentacja zmieniła średni recall Egyptian Mau o 11,11 p.p. i macro-F1 o 1,70 p.p. Recall wzrósł w 3 z 3 iteracji, pozostał bez zmian w 0, a spadł w 0.

W porównaniu z samym powielaniem zdjęć różnica średniego recall wyniosła 6,67 p.p. na rzecz augmentacji (wartość ujemna oznacza przewagę oversamplingu). To porównanie rozdziela efekt większej liczby obserwacji od efektu zmiany ich wyglądu. Budżet i źródła dodatkowych obserwacji są identyczne.

Eksperyment pokazuje, że w tym ustawieniu augmentacja może poprawić średnią wykrywalność niedoreprezentowanej klasy. Nie oznacza to gwarancji poprawy przy innym podziale, transformacjach lub klasyfikatorze.

Spełnienie celu wymaga również kontroli jakości globalnej. Średnie macro-F1 nie spadło względem bazowego.

## Walidacja i ograniczenia

Walidacja została oceniona dla każdego modelu i zapisana osobno. Nie służyła do zmieniania C ani do dobierania transformacji: parametry ustalono z góry. Testu nie wykorzystano do strojenia.

Jedna poprawnie sklasyfikowana fotografia Egyptian Mau to 3,33 p.p. recall. Przy 30 przykładach tej klasy i trzech nakładających się losowych podziałach oszacowania są niestabilne. Nie przedstawiamy testu istotności ani mocnych wniosków populacyjnych. Odchylenie między iteracjami opisuje wrażliwość na podział, a nie niezależne powtórzenia na nowych populacjach.

Kontrole wykazały rozłączność oryginałów i rodziców augmentacji wewnątrz każdego podziału. Nie eliminowano podobnych, ale nieidentycznych zdjęć ani tożsamości tych samych zwierząt. Użyto tylko jednej architektury i klasycznej augmentacji; nie wykonano eksperymentu dyfuzyjnego ani porównania z EfficientNet-B0.

## Pliki dowodowe

`results/metrics.csv` zawiera wszystkie wyniki testowe i walidacyjne, `results/summary.csv` agregaty. Katalogi każdego przebiegu zawierają pełne raporty ośmiu klas, macierze pomyłek, predykcje z prawdopodobieństwami, listę obserwacji treningowych i parametry klasyfikatora. `results/leakage_check.json` oraz `class_counts.csv` dokumentują kontrole.

![Wyniki testowe](results/comparison.png)

Przykłady: oryginały u góry, augmentacje poniżej.

![Przykłady augmentacji](results/augmentation_examples.jpg)

Źródła, licencja i protokół: `protocol.md` oraz `DATA_LICENSE.md`.

## Autor: Wiktor Wdowczyk

## Meeting 2: status realizacji

### Zakres wykonania

Archiwum Oxford-IIIT Pet zostało pobrane. Przygotowano 910 oryginalnych zdjęć, podziały, augmentacje i cechy 1,060 obrazów. Niezależny audyt Meeting 1 zakończył się sukcesem: 9 przebiegów, 18 ewaluacji oraz kontrole identyfikatorów, metryk i liczności.

Stable Diffusion 1.5 działał na CUDA 12.6, karcie NVIDIA GeForce GTX 1060 6GB, w trybie FP16 z CPU offload i attention slicing. Generowanie zostało zatrzymane po prośbie użytkownika; zachowano każdą ukończoną próbę.

| Pula | Zapisane próby | Użyteczne obrazy | Błędy | Safety block | Stan |
|---|---:|---:|---:|---:|---|
| `m2-base` | 110 | 110 | 0 | 0 | kompletna |
| `m2-guidance-2` | 11 | 11 | 0 | 0 | przerwana; pula niekompletna |
| `m2-guidance-15` | 0 | 0 | 0 | 0 | nieuruchomiona |
| `m2-steps-10` | 0 | 0 | 0 | 0 | nieuruchomiona |

Wszystkie 121 zakończonych prób mają status `ok`. Manifesty, ustawienia, obrazy i kontaktówki są w `pools/`; pliki pul nie są ignorowane przez Git. Kontaktówka guidance-2 pokazuje 11 wygenerowanych przykładów, a pozostałe miejsca są puste, bo wariant przerwano.

### Obserwacje wizualne

- Baza zawiera kilka wiarygodnych srebrnych kotów w cętki, ale również pręgowane koty domowe, nieregularne lub przesadzone wzory, zniekształcone pyski i oczy oraz zbyt ciasne kadry.
- W 11 próbkach guidance 2 widać tę samą niejednorodność: część obrazów ma cętki, ale kilka przedstawia wyraźnie pręgowane umaszczenie lub kota, którego rasy nie da się wiarygodnie rozpoznać. Pojawiają się też problemy proporcji pyska i ujęcia portretowe.
- Pozy i tła są zróżnicowane; wizualna różnorodność nie gwarantuje jednak poprawnej rasy ani przydatności treningowej.
- Są to jakościowe obserwacje z kontaktówek, nie formalna ocena ani wynik klasyfikatora. Kontaktówki: [baza](pools/m2-base/contact-sheet.jpg), [guidance 2](pools/m2-guidance-2/contact-sheet.jpg).

### Wyniki modelu i ograniczenia

Nie uruchomiono `meeting2.py train` ani `meeting2.py test`. Nie ma zatem metryk walidacyjnych lub testowych dla syntetycznych zdjęć, nie wybrano puli do przeniesienia do Meeting 3 i nie można stwierdzić, czy generowane obrazy poprawiają recall Egyptian Mau albo macro-F1. Niepełna pula guidance 2 nie powinna być używana do treningu zgodnego z protokołem. Wyniki liczbowe w poprzednich sekcjach tego pliku dotyczą wyłącznie Meeting 1.

Zakres wykonania nie spełnia pełnego checkpointu kursu: brakuje dwóch wariantów, porównań walidacyjnych, uzasadnionego wyboru i sparowanego testu. Pełny plan, parametry i ograniczenia opisuje [`protocol.md`](protocol.md).
