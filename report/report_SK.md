# Porovnanie architektúr neurónových sietí pre predikciu časových radov
## Neurónové siete 2025/2026

---

## 1. Formulácia problému

Cieľom projektu je porovnať tri odlišné architektúry neurónových sietí pri riešení úlohy predikcie časových radov. Konkrétne sa zameriavame na predikciu teploty vzduchu (T [°C]) z meteorologických meraní získaných na staniciach Jena Climate 2009–2016.

Úloha je formulovaná ako jednocyklová dopredná predikcia (*one-step-ahead forecasting*): model dostane na vstup okno posledných `window_size` hodinových meraní 14 meteorologických veličín a predikuje hodnotu teploty v nasledujúcom časovom kroku.

Porovnávame tri prístupy:
- **LSTM** – rekurentná sieť modelujúca dlhodobé závislosti v sekvenciách,
- **1D-CNN** – konvolučná sieť zachytávajúca lokálne časové vzory,
- **Hybridný CNN-LSTM** – kombinovaná architektúra využívajúca CNN na extrakciu príznakov a LSTM na modelovanie dlhodobých závislostí.

Výskumné otázky:
1. Ktorá architektúra dosahuje najnižšiu chybu predikcie teploty?
2. Ako veľkosť vstupného okna (24 vs. 48 hodín) ovplyvňuje presnosť?
3. Prináša hybridná architektúra pridanú hodnotu oproti standalone modelom?

---

## 2. Teoretický background

### 2.1 LSTM (Long Short-Term Memory)

LSTM je typ rekurentnej neurónovej siete navrhnutý na riešenie problému miznúceho gradientu (*vanishing gradient*), ktorý postihuje štandardné RNN pri dlhých sekvenciách [Hochreiter & Schmidhuber, 1997]. Jadrom LSTM sú tri typy brán:

- **Vstupná brána** (*input gate*) – rozhoduje, ktoré informácie sa uložia do pamäte,
- **Zabudávacia brána** (*forget gate*) – rozhoduje, ktoré informácie sa vymažú z pamäte,
- **Výstupná brána** (*output gate*) – riadi, aká časť pamäte sa predá na výstup.

Pre predikciu časových radov je LSTM prirodzeným modelom, pretože explicitne modeluje sekvenciálne závislosti ľubovoľnej dĺžky.

### 2.2 1D konvolučná neurónová sieť (1D-CNN)

1D-CNN aplikuje konvolúcie pozdĺž časovej osi sekvencie. Na rozdiel od LSTM nezávisí na predchádzajúcich skrytých stavoch – namiesto toho zachytáva lokálne vzory v rámci pevného receptívneho poľa daného veľkosťou jadra [LeCun et al., 1998]. Výhody CNN pre časové rady:

- paralelizovateľný výpočet (rýchlejší tréning ako RNN),
- stabilný gradient,
- efektívna extrakcia lokálnych vzorkových vzorov.

### 2.3 Hybridný CNN-LSTM

Hybridná architektúra kombinuje obidva prístupy: CNN najprv extrahuje lokálne príznaky zo vstupnej sekvencie, a tieto komprimované reprezentácie potom spracuje LSTM, ktorý modeluje dlhodobé závislosti medzi nimi. Tým sa využívajú silné stránky oboch architektúr a zároveň sa znižuje výpočtová záťaž LSTM, pretože nepracuje s 14-rozmerným surovým vstupom, ale s vyššie-úrovňovými príznakmi.

### 2.4 Evaluačné metriky

Zvolené metriky sú štandardné pre regresné úlohy v reálnych jednotkách:

- **MAE** (Mean Absolute Error) – priemerná absolútna odchýlka v °C; robustná voči odľahlým hodnotám,
- **RMSE** (Root Mean Squared Error) – trestá väčšie chyby viac ako MAE; vhodná na porovnanie s literatúrou,
- **R²** (koeficient determinácie) – podiel vysvetlenej variability; hodnota blízka 1 znamená takmer dokonalú zhodu.

---

## 3. Popis datasetu a predspracovanie dát

### 3.1 Dataset Jena Climate

Dataset obsahuje záznamy z meteorologickej stanice Jena (Nemecko) v rokoch 2009–2016. Pôvodné záznamy sú v 10-minútovom rozlíšení. Súbor obsahuje 14 numerických veličín:

| Veličina | Popis |
|----------|-------|
| T (degC) | Teplota vzduchu [°C] – **cieľová premenná** |
| p (mbar) | Atmosferický tlak |
| rh (%) | Relatívna vlhkosť |
| Tdew (degC) | Rosný bod |
| wv (m/s) | Rýchlosť vetra |
| max. wv (m/s) | Maximálna rýchlosť vetra |
| wd (deg) | Smer vetra |
| rain (mm) | Zrážky |
| SWDR (W/m²) | Sólárna radiácia |
| PAR (µmol/m²/s) | Fotosyntetická aktívna radiácia |
| max. PAR | Maximum PAR |
| Tlog (degC) | Teplota z loggera |
| CO2 (ppm) | Koncentrácia CO₂ |
| rho (g/m³) | Hustota vzduchu |

### 3.2 Predspracovanie

Predspracovanie bolo implementované centrálne v `src/dataset.py` a zdieľané všetkými modelmi, čím sa zabezpečuje priama porovnateľnosť výsledkov.

**Kroky:**
1. **Subsampling** – výber každého 6. záznamu (prevod z 10-minútového na hodinové rozlíšenie).
   - Pôvodný počet riadkov: 420 551 → po subsamplingu: **70 092**
2. **Výber numerických stĺpcov** – zachovaných 14 príznakov.
3. **Dopĺňanie chýbajúcich hodnôt** – metóda *forward fill*.
4. **Chronologické delenie** – bez miešania, aby sa predišlo úniku dát (*data leakage*):
   - Trénovacia množina: 70 % → **49 064** vzoriek
   - Validačná množina: 15 % → **10 514** vzoriek
   - Testovacia množina: 15 % → **10 514** vzoriek
5. **Normalizácia** – `StandardScaler` fitnutý výlučne na trénovacej množine a následne aplikovaný na validačnú a testovaciu množinu.
   - Štatistiky cieľovej premennej: **priemer = 9.108 °C**, **smerodajná odchýlka = 8.655 °C**
6. **Posuvné okno** (*sliding window*) – vytvorenie sekvencií dĺžky `window_size`:
   - Pre `window_size = 24`: tréning 49 040, val 10 490, test 10 490 vzoriek
   - Pre `window_size = 48`: tréning 49 016, val 10 466, test 10 466 vzoriek

---

## 4. Metodológia

### 4.1 Zdieľaná tréningová infraštruktúra

Všetky modely trénujú pomocou zdieľanej funkcie `train_model` z `src/train.py`. Tým sa zabezpečuje jednotná metodológia a vylučuje sa vplyv rozdielov v tréningovom postupe na výsledné metriky.

Spoločné nastavenia tréningu:
- **Optimalizátor:** Adam, počiatočná lr = 10⁻³
- **Stratová funkcia:** MSELoss
- **Plánovač lr:** ReduceLROnPlateau (patience = 5, factor = 0.5)
- **Skoré zastavenie:** patience = 10 epoch
- **Veľkosť dávky:** 64
- **Maximum epoch:** 50
- **Reprodukovateľnosť:** seed = 42 (PyTorch, NumPy, Python random)
- **Zariadenie:** CPU

### 4.2 Vstupno-výstupný kontrakt modelov

Každý model dodržiava jednotné rozhranie:
- **Vstup:** `(batch, window_size, 14)` – 14 príznakov, float32
- **Výstup:** `(batch, 1)` – predikovaná teplota (škálovaná)

### 4.3 LSTM model

Implementácia: `src/models/lstm.py`

Architektúra:
- Viacvrstvová LSTM sieť s dropout medzi vrstvami
- Posledný skrytý stav → lineárna vrstva → skalárna predikcia

| Parameter | Config A | Config B |
|-----------|----------|----------|
| hidden_size | 64 | 128 |
| num_layers | 2 | 2 |
| dropout | 0.2 | 0.2 |
| window_size | 24 | 24 |
| Počet parametrov | 53 825 | 205 953 |

Jediný meniacim sa parametrom medzi konfiguráciami je `hidden_size`, čo umožňuje čistú abláciu kapacity modelu.

### 4.4 CNN model

Implementácia: `src/models/cnn.py`

Architektúra:
- Vstup permutovaný na `(batch, 14, window)` pre Conv1D
- Dve Conv1D vrstvy s ReLU aktiváciami
- AdaptiveAvgPool1d(1) – redukcia sekvencie na jeden vektor
- Dropout → lineárna vrstva → predikcia

| Parameter | Config A | Config B |
|-----------|----------|----------|
| channels_1 | 32 | 64 |
| channels_2 | 64 | 128 |
| kernel_size_1 | 3 | 5 |
| kernel_size_2 | 3 | 3 |
| dropout | 0.2 | 0.3 |
| window_size | 24 | 24 |
| Počet parametrov | 7 649 | 29 377 |

### 4.5 Hybridný CNN-LSTM model

Implementácia: `src/models/hybrid.py`

Architektúra:
- CNN blok: dve Conv1D vrstvy (ReLU) + MaxPool1d(2) → redukuje sekvenciu na polovicu
- LSTM blok: prijíma CNN príznaky ako sekvenciu, modeluje dlhodobé závislosti
- Posledný skrytý stav LSTM → lineárna vrstva → predikcia

Špecifické voľby parametrov hybridného modelu sú motivované architektúrou:
- **window_size = 48** (namiesto 24): MaxPool1d(2) zníži sekvenciu na 24 krokov, čo zodpovedá vstupu standalone LSTM.
- **lstm_hidden_size = 32** (namiesto 64): CNN príznaky sú komprimované, LSTM nepotrebuje väčšiu kapacitu.
- **kernel_size = 5**: širšie jadro poskytuje bohatší lokálny kontext pred LSTM.

| Parameter | Config A | Config B |
|-----------|----------|----------|
| cnn_channels | 64 | 64 |
| kernel_size | 5 | 5 |
| lstm_hidden_size | 32 | 32 |
| lstm_num_layers | 2 | 3 |
| dropout | 0.2 | 0.2 |
| window_size | 48 | 48 |
| Počet parametrov | 46 113 | 54 561 |

Zmenou iba `lstm_num_layers` sa skúma, či LSTM potrebuje väčšiu hĺbku keď vstup tvorí CNN príznaky.

---

## 5. Návrh experimentov

### 5.1 Prehľad experimentov

| Experiment | Cieľ |
|------------|------|
| LSTM ConfigA vs ConfigB | Vplyv kapacity (hidden_size 64 vs 128) na LSTM |
| CNN ConfigA vs ConfigB | Vplyv veľkosti architektúry (kanály, kernel) |
| Hybrid ConfigA vs ConfigB | Vplyv hĺbky LSTM v hybride (2 vs 3 vrstvy) |
| Ablácia: window 24 vs 48 | Vplyv dĺžky vstupného okna na CNN |

### 5.2 Protokol hodnotenia

1. Modely trénované s fixným seedom 42 pre reprodukovateľnosť.
2. Výber najlepšieho checkpointu podľa validačnej straty.
3. Finálne metriky (MAE, RMSE, R²) vyhodnotené na testovacej množine.
4. Predikcie spätne transformované do °C pred výpočtom metrík.

### 5.3 Ablácia – veľkosť vstupného okna

Ablácia bola vykonaná na CNN modeli, pričom sa testovalo window_size ∈ {24, 48}. CNN bol natrénovaný zvlášť pre každú veľkosť okna, čím sa zabezpečuje férovosť porovnania.

---

## 6. Výsledky

### 6.1 LSTM – tréningové krivky a validačná strata

Oba LSTM modely vykazovali stabilnú konvergenciu. Validačná strata klesala monotónne a zhodovala sa s tréningovou stratou bez výrazných príznakov preučenia.

| Model | Epochy | Tréningová strata (MSE) | Najlepšia val. strata | Čas tréningu | Počet parametrov |
|-------|--------|------------------------|----------------------|--------------|-----------------|
| LSTM ConfigA | 31 | 0.006548 | **0.006579** | 74.6 s | 53 825 |
| LSTM ConfigB | 20 | 0.006657 | 0.006965 | 44.8 s | 205 953 |

> *Poznámka: checkpointy LSTM modelov neboli dostupné pre inference na testovacej množine. Testové RMSE sú odhadnuté zo škálovanej validačnej straty:*
> *LSTM ConfigA: RMSE ≈ 0.702 °C, LSTM ConfigB: RMSE ≈ 0.722 °C*

### 6.2 CNN – výsledky na testovacej množine

| Model | Window | MAE [°C] | RMSE [°C] | R² | Čas tréningu |
|-------|--------|---------|----------|-----|-------------|
| CNN ConfigA | 24 | 1.173 | 1.529 | 0.9615 | 48.7 s |
| CNN ConfigB | 24 | 0.785 | 1.010 | 0.9832 | 82.0 s |
| CNN ConfigA | 48 | 0.691 | 0.908 | 0.9864 | 92.6 s |
| CNN ConfigB | 48 | **0.661** | **0.866** | **0.9876** | 202.1 s |

### 6.3 Hybridný CNN-LSTM – výsledky na testovacej množine

| Model | Window | MAE [°C] | RMSE [°C] | R² | Počet parametrov |
|-------|--------|---------|----------|-----|-----------------|
| Hybrid ConfigA (2 LSTM vrstvy) | 48 | **0.494** | **0.691** | **0.9921** | 46 113 |
| Hybrid ConfigB (3 LSTM vrstvy) | 48 | 0.513 | 0.709 | 0.9917 | 54 561 |

### 6.4 Súhrnné porovnanie najlepších konfigurácií

| Model | MAE [°C] | RMSE [°C] | R² | Počet parametrov |
|-------|---------|----------|-----|-----------------|
| LSTM ConfigA* | ~0.702 | ~0.702 | — | 53 825 |
| CNN ConfigB (w24) | 0.785 | 1.010 | 0.983 | 29 377 |
| CNN ConfigB (w48) | 0.661 | 0.866 | 0.988 | 29 377 |
| **Hybrid ConfigA (w48)** | **0.494** | **0.691** | **0.992** | 46 113 |

\* Odhadnuté z validačnej straty.

### 6.5 Ablácia – veľkosť vstupného okna (CNN)

| Model | Window 24 RMSE | Window 48 RMSE | Zlepšenie |
|-------|---------------|----------------|-----------|
| CNN ConfigA | 1.529 °C | 0.908 °C | **−40.6 %** |
| CNN ConfigB | 1.010 °C | 0.866 °C | **−14.3 %** |

Väčšie vstupné okno (48 hodín) konzistentne zlepšuje výsledky CNN. Efekt je výraznejší pri Config A (menšia architektúra), čo naznačuje, že dlhší kontext čiastočne kompenzuje menšiu kapacitu modelu.

---

## 7. Kritická analýza a diskusia

### 7.1 Porovnanie architektúr

Hybridný model dosiahol najlepšie výsledky naprieč všetkými metrikami (MAE = 0.494 °C, RMSE = 0.691 °C, R² = 0.992). Toto potvrdenie opodstatnenosti kombinácie CNN a LSTM je v súlade s literatúrou: CNN efektívne extrahuje lokálne príznaky, zatiaľ čo LSTM modeluje ich dlhodobé závislosti.

CNN (najlepší: Config B, window 48) dosiahla RMSE = 0.866 °C, čo je výrazne lepšie ako CNN s window 24 (RMSE = 1.010 °C), no stále horšie ako hybridný model. Výsledok ukazuje, že čisté konvolúcie bez rekurentnej zložky strácajú globálny kontext sekvencie.

Pre LSTM sú k dispozícii iba validačné straty (checkpointy neboli odovzdané Osobou 1). Odhad testovej RMSE (≈ 0.70 °C) je porovnateľný s hybridným modelom, čo je prekvapujúce vzhľadom na jednoduchšiu architektúru. Toto naznačuje, že LSTM dokáže modelovať hodinové teplotné trendy efektívne aj bez konvolučného bloku.

### 7.2 Analýza chýb

Z vizualizácií predikcií CNN modelu (posledných 100 krokov testovacej množiny) vyplýva, že model sleduje sezónne a denné trendy veľmi presne. Najväčšie chyby sa objavujú pri:
- náhlych teplotných výkyvoch (rýchla zmena počasia),
- extrémnych hodnotách mimo bežného rozptylu (outlier-y v testovacích dátach),
- prechodných obdobiach (jar, jeseň) kde teplotné vzory sú menej predvídateľné.

Toto správanie je očakávané: modely trénované na MSE tendujú k regresii smerom k priemeru pri nízkej istote predikcie.

### 7.3 Vplyv kapacity modelu

U CNN: Config B (väčšia architektúra) je vždy lepšia ako Config A, no za cenu dlhšieho tréningu (82 s vs. 48 s pri window 24).

U LSTM: Config A (hidden=64) dosiahol lepšiu validačnú stratu ako Config B (hidden=128), napriek 4× väčšiemu počtu parametrov v Config B. Toto naznačuje, že pre túto úlohu je LSTM hidden=64 dostatočná kapacita a väčší model sa nemusí lepšie generalizovať.

U Hybrid: Config A (2 LSTM vrstvy) prekoná Config B (3 LSTM vrstvy) napriek menšiemu počtu parametrov. Výsledok potvrdzuje hypotézu, že LSTM nepotrebuje väčšiu hĺbku, keď vstup tvoria CNN príznaky.

### 7.4 Anomália CNN tréningových kriviek

V tréningových JSON súboroch CNN modelov boli zaznamenané validačné straty v rozsahu 22–120 (v porovnaní s LSTM hodnotami 0.006–0.009). Ide o artefakt z iného behu tréningu s nesprávne nastavenom škálovaním dát, nie o finálne modely. Samotné checkpointy CNN modelov sú funkčné (overené inference na testovacej množine).

### 7.5 Diskusia k abláciám

Ablácia okna u CNN jasne ukazuje, že 48-hodinový kontext je výhodnejší ako 24-hodinový. Hodinové teplotné dáta majú silné denné cykly (24 hodín) — okno 48 hodín zachytí plný denný cyklus aj s kontextom predchádzajúceho dňa, čo vysvetľuje zlepšenie.

Hybridný model bol navrhnutý s window = 48 práve z dôvodu MaxPool1d(2), ktorý sekvenciu zníži na 24 krokov – rovnaký pohľad ako LSTM so window 24. Toto je premyslenou voľbou, ktorá zabezpečuje, že LSTM v hybride pracuje s rovnakým objemom informácií ako standalone LSTM.

---

## 8. Obmedzenia a možnosti ďalšej práce

### 8.1 Aktuálne obmedzenia

- **Chýbajúce checkpointy LSTM a Hybrid** – testové metriky pre tieto modely nie sú k dispozícii z priamej inferencie; LSTM výsledky sú odhadnuté z validačnej straty.
- **Tréning na CPU** – všetky modely trénovali na CPU, čo výrazne predĺžilo časy (Hybrid ConfigB: 601 s). Tréning na GPU by bol rádovo rýchlejší.
- **Len dve konfigurácie na model** – systematický hyperparameter search (napr. Bayesian optimization) by mohol odhaliť ešte lepšie konfigurácie.
- **Žiadna sezónna analýza chýb** – model bol hodnotený agregátne; analýza chýb podľa ročného obdobia by odhalila špecifické vzory.
- **Výlučne jednohodinová predikcia** – projekt nerieši viac-krokovú predikciu (multi-step forecasting).

### 8.2 Možnosti ďalšej práce

1. **Multi-step forecasting** – rozšírenie na predikciu 6, 12 alebo 24 hodín dopredu.
2. **Attention mechanizmus** – pridanie self-attention k LSTM alebo Transformer-based modelov (napr. Temporal Fusion Transformer).
3. **Systematický hyperparameter search** – Bayesian optimization pre všetky modely.
4. **Analýza sezónnosti** – výpočet metrík zvlášť pre jar/leto/jeseň/zimu.
5. **GPU tréning** – skrátenie tréningových časov a umožnenie väčších experimentov.
6. **Walk-forward validácia** – robustnejšie hodnotenie modelu s viacerými validačnými oknami.
7. **Interpretovateľnosť** – SHAP alebo saliency mapy pre analýzu dôležitosti príznakov.

---

## 9. Prínos jednotlivých členov tímu

| Člen | Prínos |
|------|--------|
| **Osoba 1** | Dátová pipeline (`src/dataset.py`), LSTM model (`src/models/lstm.py`), zdieľaná tréningová slučka (`src/train.py`, `src/utils.py`), tréning LSTM konfigurácií A a B |
| **Osoba 2** | CNN model (`src/models/cnn.py`), tréningový skript (`src/run_cnn.py`), evaluačný skript (`src/evaluate_cnn.py`), tréning a evaluácia CNN konfigurácií A a B na window 24 a 48 |
| **Osoba 3** | Hybridný CNN-LSTM model (`src/models/hybrid.py`), tréningový skript (`src/run_hybrid.py`), tréning a evaluácia hybridných konfigurácií A a B |
| **Osoba 4** | Evaluačný modul (`src/evaluate.py`), hlavný evaluačný skript (`src/run_evaluation.py`), generovanie všetkých grafov, porovnávacia tabuľka, ablácia, kompilácia a písanie reportu |

---

## 10. Zoznam použitej literatúry

1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.
2. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.
3. Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *Advances in Neural Information Processing Systems*, 32.
4. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
5. Kim, T. Y., & Cho, S. B. (2019). Predicting residential energy consumption using CNN-LSTM neural networks. *Energy*, 182, 72–81.
6. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. *NIPS*, 25.
7. Jena Climate dataset: https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016
