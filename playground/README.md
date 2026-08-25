# Eksperimentalni notebookovi

Folder `playground` sadrži reproduktivni Google Colab tok projekta. Notebookovi
su numerisani prema redosledu izvršavanja i zajedno vode od provere izvornog
LipNet modela do treninga, robustnosti i naprednog CTC dekodiranja.

## Redosled pokretanja

| Notebook | Sadržaj | Okruženje |
|---|---|---|
| [`00_faza_0_upstream_restart.ipynb`](00_faza_0_upstream_restart.ipynb) | pinovanje VIPL izvora, licenca i inventar | CPU |
| [`01_faza_1_grid_parity.ipynb`](01_faza_1_grid_parity.ipynb) | poređenje izvornog i lokalnog GRID inference-a | GPU |
| [`02_faza_2_ai_speak_preprocessing.ipynb`](02_faza_2_ai_speak_preprocessing.ipynb) | AI-SPEAK MP4 → poravnati mouth JPEG frejmovi i QA | GPU |
| [`03_faza_3_serbian_dataset.ipynb`](03_faza_3_serbian_dataset.ipynb) | srpski vokabular, Dataset i speaker-disjoint split | CPU |
| [`04_faza_4_transfer_ctc_smoke.ipynb`](04_faza_4_transfer_ctc_smoke.ipynb) | transfer VIPL težina i CTC backward provera | GPU |
| [`05_faza_5_baseline_finetuning.ipynb`](05_faza_5_baseline_finetuning.ipynb) | fine-tuning, resume i izbor najboljeg checkpoint-a | GPU |
| [`06_faza_6_robustness_experiments.ipynb`](06_faza_6_robustness_experiments.ipynb) | rezolucija, blur i pomeranje mouth crop-a | GPU |
| [`07_faza_7_decoder_search.ipynb`](07_faza_7_decoder_search.ipynb) | greedy, prefix beam search, 5-gram LM i bootstrap | GPU/CPU |
| [`08_faza_8_konsolidovani_notebook.ipynb`](08_faza_8_konsolidovani_notebook.ipynb) | GPU demonstracija, konsolidovani rezultati, konfuzije i figure za izveštaj | GPU |

Faze 3–7 koriste već pripremljenu `ai_speak_lip.zip` arhivu, pa preprocessing
ne mora ponovo da se izvršava pri svakoj evaluaciji.

## Rezultat faze 07

Notebook `07_faza_7_decoder_search.ipynb` je izvršen do kraja na NVIDIA L4.
Greedy kontrola se tačno poklopila sa rezultatom faze 05, a validation skup je
izabrao sledeće zaključane konfiguracije:

- prefix beam bez jezičkog modela: beam width 50;
- prefix beam sa train-only karakternim 5-gram modelom: beam width 50,
  `α = 1,0` i `β = 0,5`.

| Dekoder | Validation WER | Test WER | Test CER |
|---|---:|---:|---:|
| Greedy | 49,72% | 45,25% | 18,24% |
| Prefix beam bez LM-a | 49,69% | 44,88% | 18,10% |
| Prefix beam + 5-gram LM | **45,40%** | **41,20%** | **14,70%** |

Jezički model je fitovan nad svih 2.877 train transkripata, bez korišćenja
validation ili test transkripata. Notebook je na Google Drive sačuvao
`decoder_results_v1.json`, `decoder_predictions_v1.json` i grafikon
`decoder_metrics_v1.png`. Sanitizovane kopije oba JSON artefakta nalaze se u
[`docs/results`](../docs/results/README.md); rezultatni JSON uključuje paired
bootstrap intervale i analizu grešaka po pozicijama.

## Generisanje notebookova

Notebookovi su izlaz skripte
[`scripts/build_phase_notebooks.py`](../scripts/build_phase_notebooks.py). Da bi
kod i notebookovi ostali usklađeni, njihove ćelije se ne menjaju ručno.

```powershell
uv run python scripts/build_phase_notebooks.py
```

## Ulazi i rezultati

Notebookovi montiraju Google Drive za velike ulaze i checkpoint-e. Mali,
sanitizovani rezultati koji se koriste u finalnom radu nalaze se u
[`docs/results`](../docs/results/README.md). Detaljne podrazumevane Drive putanje
navedene su u [glavnom README-u](../README.md#podaci-i-artefakti).

## Rezultat faze 08

Notebook 08 učitava i međusobno proverava JSON artefakte Faza 03–07, proverava
SHA-256 zaključanog `best.pt` checkpoint-a i obavezno izvršava predikciju za test
primer indeksa 42 na GPU-u. Jezički model se fituje samo nad 2.877 train
transkripata; validation i test transkripti služe isključivo za evaluaciju.

Notebook je izvršen do kraja na NVIDIA L4: svih 18 kodnih ćelija sadrži output,
a završna provera prijavljuje `PASS` za checkpoint, GPU demonstraciju, metrike,
bootstrap intervale i figure. Na demonstracionom primeru 42 beam+5-gram izlaz
tačno reprodukuje referencu `potvrdi v gore ž nedelja šest`, dok greedy izlaz
izostavlja oba izolovana slova.

Pored prikaza u notebooku, sedam PNG figura čuva se u
`MyDrive/LipNet/phase8_report/`: podela skupa, mouth frejmovi demonstracije,
robustnost, poređenje dekodera sa bootstrap intervalima, tačnost slotova, četiri
normalizovane matrice konfuzije i najčešće zamene izolovanih slova. Matrica iz
teorijske prezentacije ostaje ilustracija originalnog LipNet rada na GRID-u i ne
predstavlja rezultat ovog projekta.

Kopije ovih figura nalaze se u [`report/assets`](../report/README.md#slike), uz
dodatni prikaz istorije treninga. Finalni
[`PDF`](../report/finalni-izvestaj-lipnet.pdf) i
[`HTML`](../report/finalni-izvestaj-lipnet.html) izveštaj takođe su verzionisani.

Generator pravi samo ovaj notebook naredbom:

```powershell
uv run python scripts/build_phase_notebooks.py --phase 8
```

Naredbu za generisanje koristiti samo kada se menja izvor notebooka: ona pravi
novu, neizvršenu verziju, pa bi postojeći Colab output trebalo sačuvati ili potom
ponovo izvršiti notebook. Trenutna verzija u ovom folderu predstavlja završeni
Colab run korišćen za finalni izveštaj.
