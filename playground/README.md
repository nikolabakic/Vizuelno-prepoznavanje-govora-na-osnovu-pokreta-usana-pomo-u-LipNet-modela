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

Faze 3–7 koriste već pripremljenu `ai_speak_lip.zip` arhivu, pa preprocessing
ne mora ponovo da se izvršava pri svakoj evaluaciji.

## Generisanje notebookova

Notebookovi su izlaz skripte
[`scripts/build_phase_notebooks.py`](../scripts/build_phase_notebooks.py). Da bi
kod i notebookovi ostali usklađeni, njihove ćelije se ne menjaju ručno.

```powershell
uv run python scripts/build_phase_notebooks.py
```

Test `test_phase_notebook_sources_match_generator` proverava da verzionisani
notebookovi odgovaraju generatoru.

## Ulazi i rezultati

Notebookovi montiraju Google Drive za velike ulaze i checkpoint-e. Mali,
sanitizovani rezultati koji se koriste u finalnom radu nalaze se u
[`docs/results`](../docs/results/README.md). Detaljne podrazumevane Drive putanje
navedene su u [glavnom README-u](../README.md#podaci-i-artefakti).
