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

Przygotowano 910 oryginalnych zdjęć i istniejący cache cech; niezależny audyt Meeting 1 zakończył się sukcesem. Dla Meeting 2 wygenerowano pule Stable Diffusion 1.5 na GPU w Colabie. Następnie lokalnie wyekstrahowano cechy ResNet-18, wykonano pięć sparowanych przebiegów walidacyjnych dla każdej puli i przeprowadzono test wyłącznie dla puli wybranej na podstawie walidacji.

| Pula | Zapisane próby | Użyteczne obrazy | Błędy | Safety block | Stan |
|---|---:|---:|---:|---:|---|
| `m2-base` | 110 | 110 | 0 | 0 | kompletna |
| `m2-guidance-2` | 110 | 109 | 0 | 1 | kompletna; 109 obrazów dopuszczonych do użycia |
| `m2-guidance-15` | 110 | 110 | 0 | 0 | kompletna |
| `m2-steps-10` | 110 | 110 | 0 | 0 | kompletna |

Łącznie: 440 prób, 439 użytecznych obrazów, jeden safety block i zero błędów generowania. Obrazy, manifesty, ustawienia i kontaktówki znajdują się w `data/pools/` i są uwzględnione w repozytorium.

### Obserwacje wizualne

- Większość wygenerowanych obrazów jest istotnie zdeformowana albo zawiera poważnie uszkodzone cechy: zniekształcone pyski i oczy, nieprawidłową liczbę lub układ kończyn, zaburzone proporcje ciała, nienaturalne wzory sierści albo niepoprawne połączenia części ciała. W wielu przypadkach wada obejmuje centralny obiekt, a nie tylko drobny szczegół czy kadr, przez co obraz nie wygląda jak wiarygodna fotografia kota.
- Część próbek ma poprawniejszą sylwetkę lub rozpoznawalne umaszczenie Egyptian Mau, ale nie zmienia to ogólnie niskiej jakości wizualnej pul. Występują też koty pręgowane lub o niejednoznacznej rasie oraz ciasne kadry. Zgodność z promptem jest ograniczona; ocena wizualna jest jakościowa i wskazuje, że większość obrazów nie nadaje się do użycia jako realistyczne przykłady treningowe bez selekcji lub odfiltrowania.
- Kontaktówki: [baza](data/pools/m2-base/contact-sheet.jpg), [guidance 2](data/pools/m2-guidance-2/contact-sheet.jpg), [guidance 15](data/pools/m2-guidance-15/contact-sheet.jpg), [10 kroków](data/pools/m2-steps-10/contact-sheet.jpg).

### Wyniki modelu i ograniczenia

Klasyfikator pozostał zgodny z metodą tego repo: zamrożony ResNet-18 ImageNet i regresja logistyczna. Każda pula wnosi 55 obrazów do danego splitu; ewaluacja objęła pięć sparowanych seedów. Wyniki walidacji (średnia ± odchylenie standardowe):

| Warunek | Recall Egyptian Mau | Macro-F1 | Accuracy |
|---|---:|---:|---:|
| Baseline tylko na danych rzeczywistych | 69.00% ± 9.62% | 86.97% ± 1.79% | 87.00% ± 1.90% |
| `m2-base` | 70.00% ± 10.61% | 86.59% ± 1.76% | 86.62% ± 1.91% |
| `m2-guidance-2` | **87.00% ± 5.70%** | **87.93% ± 1.03%** | **88.00% ± 1.03%** |
| `m2-guidance-15` | 66.00% ± 12.94% | 85.78% ± 1.95% | 85.88% ± 2.10% |
| `m2-steps-10` | 73.00% ± 7.58% | 86.84% ± 1.50% | 86.88% ± 1.65% |

Na podstawie wyłącznie walidacji i kontaktówek wybrano `m2-guidance-2`: uzyskał najwyższy średni recall Mau i macro-F1. Jego pięć splitów testowych dało recall **83.33% ± 4.08%**, macro-F1 **86.81% ± 1.19%** i accuracy **86.75% ± 1.26%**. Sparowany baseline testowy osiągnął odpowiednio **62.67% ± 9.25%**, **84.13% ± 1.59%** i **84.17% ± 1.67%**. Test nie był użyty do wyboru puli.

To wynik eksperymentu na ustalonych splitach i z klasyfikatorem tego repozytorium; nie dowodzi ogólnej przewagi modeli ani nie jest bezpośrednio porównywalny z template'owym fine-tuningiem EfficientNet. W szczególności kontaktówki wskazują, że jakość poszczególnych syntetycznych zdjęć jest nierówna. Pełny raport, predykcje, metryki sparowane i wykresy pozostają w lokalnym, ignorowanym przez Git `results/meeting2/`. Protokół i jego ograniczenia opisuje [`protocol.md`](protocol.md).
