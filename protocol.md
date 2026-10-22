# Protokół eksperymentu

## Pytanie i zakres
Czy klasyczna augmentacja poprawia rozpoznawanie niedoreprezentowanej rasy Egyptian Mau względem treningu niezbalansowanego i prostego powielania obserwacji, bez pogorszenia macro-F1? To trzy kontrole z meeting-1, bez generowania dyfuzyjnego. Protokół i parametry ustalono przed odczytaniem wyników testowych.

## Dane i podziały
Oxford-IIIT Pet, osiem ras. Jednorazowo losujemy 120 zdjęć każdej z siedmiu ras większościowych i 70 Egyptian Mau: 910 oryginalnych plików JPEG. Ziarno wyboru puli: 20261018. Kandydaci są sortowani po nazwie, tasowani niezależnie w obrębie rasy jednym generatorem NumPy. Uszkodzone obrazy i identyczne obrazy po dekodowaniu są pomijane. Oryginały nie są rekompresowane. `selection.csv` zawiera etykiety i SHA-256 plików oraz pikseli; `selection_info.json` zawiera SHA-256 archiwum źródłowego.

Dla trzech iteracji używamy ziaren 1729, 2718, 31415. W każdej rasie losowa permutacja tej samej puli wyznacza kolejno 30 zdjęć testowych, 20 walidacyjnych i pozostałe treningowe.

| Grupa | Trening bazowy | Walidacja | Test | Razem oryginałów |
|---|---:|---:|---:|---:|
| Każda z 7 ras większościowych | 70 | 20 | 30 | 120 |
| Egyptian Mau | 20 | 20 | 30 | 70 |
| Łącznie | 510 | 160 | 240 | 910 |

Każda iteracja ma własny manifest; wewnątrz niej wszystkie trzy metody mają identyczne oryginały, walidację i test. Pomiędzy iteracjami zdjęcia mogą zmieniać rolę: to powtarzany losowy holdout, a nie trzy rozłączne zbiory ani walidacja krzyżowa.

## Trzy warunki
1. **baseline**: 510 obserwacji treningowych, w tym 20 Egyptian Mau.
2. **oversampling**: 510 oryginałów i 50 powtórzeń Egyptian Mau, razem 560 obserwacji; 70 na klasę. Kopie są wpisami manifestu, bez zbędnych kopii plików.
3. **augmentation**: te same 510 oryginałów i 50 zapisanych obrazów PNG utworzonych z 20 treningowych Egyptian Mau; razem 560 obserwacji. Augmentacja jest jednorazowa (offline), nie losowana na nowo przy każdej epoce.

Dobór źródeł dla obu metod jest identyczny: dwie permutacje 20 oryginałów i pierwszych 10 elementów trzeciej. Każdy oryginał ma 2 lub 3 dodatkowe obserwacje. Ziarno wynosi ziarno podziału + 100000. Nie stosujemy dodatkowych wag klas.

Transformacje: losowy kadr obejmujący 85–98% szerokości i wysokości, przeskalowanie do 256×256, obrót od −15° do +15° z odbiciowym dopełnieniem, odbicie poziome w 25 z 50 obrazów, mnożnik jasności i kontrastu niezależnie w [0,85; 1,15]. Parametry i rodzice zapisani w `added_N.json`. Kadr i skalowanie mogą zmieniać proporcje i usuwać szczegóły; jest to ograniczenie tego konkretnego wariantu augmentacji.

## Reprezentacja i uczenie
PyTorch/torchvision, ResNet-18 z `ResNet18_Weights.IMAGENET1K_V1`. Cały ekstraktor jest zamrożony w trybie `eval`, warstwa końcowa zastąpiona Identity. Standardowe przetwarzanie torchvision: resize krótszego boku do 256, centralny kadr 224×224, normalizacja ImageNet. Każdy obraz daje 512 cech. Brak dostrajania sieci i aktualizacji BatchNorm. Wagi nie były trenowane na niniejszym zadaniu; ich pochodzenie to ImageNet.

Cechy standaryzujemy przez StandardScaler dopasowany tylko na 510 oryginałach treningowych danej iteracji, wspólny dla trzech metod. Klasyfikator: wieloklasowa regresja logistyczna scikit-learn, regularyzacja L2, C=1, solver lbfgs, tol=1e-7, max_iter=3000. Parametry są z góry stałe. Walidacja służy diagnostyce, a nie wyborowi hiperparametrów. Nie stosujemy early stopping ani strojenia po teście. Jest to zgodny z zaleceniem strony lekki klasyfikator na zamrożonych cechach, a nie uczenie CNN od zera. Wpływ zwiększenia liczby obserwacji na funkcję celu solvera jest częścią porównywanych metod.

## Ocena
Główna metryka: recall Egyptian Mau = TP/(TP+FN). Kontrolna: macro-F1, nieważona średnia F1 ośmiu klas. Dodatkowo accuracy, precision i F1 Egyptian Mau, pełne raporty wszystkich klas, prawdopodobieństwa i macierze pomyłek (wiersz = prawda, kolumna = predykcja). Wszystkie metryki w CSV są w skali 0–1. Test ma 30 Egyptian Mau, więc zmiana jednej poprawnej decyzji zmienia recall o 3,33 punktu procentowego.

Raportujemy każdy z dziewięciu przebiegów, średnie i odchylenia standardowe próby (ddof=1) oraz sparowane różnice w ramach iteracji. Nie wybieramy korzystniejszego podziału. Trzy powtórzenia nie wystarczają do mocnych wniosków o istotności statystycznej; zbiory testowe pomiędzy iteracjami mogą się nakładać.

## Kontrole i odtwarzalność
Automatyczne asercje sprawdzają liczności, unikalność ID i hashy pikseli oryginałów, zgodność SHA-256 plików, rozłączność podzbiorów w obrębie iteracji i przynależność rodziców augmentacji do treningu. Kontrola nie wyklucza bardzo podobnych zdjęć ani wielu zdjęć tego samego zwierzęcia. Cechy testowe można obliczyć wspólnym zamrożonym ekstraktorem: nie dopasowuje się on do danych i nie korzysta ze statystyk testu. Skalowanie i regresja korzystają wyłącznie z treningu.

Cache cech zawiera listę ścieżek i SHA-256 wejść; nieaktualny cache jest odrzucany. CPU, cztery wątki PyTorch, deterministyczne operacje; zapisane wersje bibliotek w `results/environment.json`. Różne wersje bibliotek lub platformy mogą dać drobne różnice numeryczne. Reprodukcja z cache nie wymaga pobierania wag. Regeneracja cech pobierze około 45 MB wag; odtworzenie wyboru puli wymaga oryginalnego archiwum około 755 MiB.

## Źródła
- Zadanie i zalecenia: https://patrykbart.faculty.wmi.amu.edu.pl/teaching/synthetic-data-generation-with-diffusion-models/meeting-1/ (odczyt 18.10.2026). Przyjęty protokół wykorzystuje 20 zdjęć treningowych mniejszości i budżet 50 dodatkowych obserwacji.
- Dane: https://www.robots.ox.ac.uk/~vgg/data/pets/
- Model: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
- O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar, „Cats and Dogs”, CVPR 2012.
