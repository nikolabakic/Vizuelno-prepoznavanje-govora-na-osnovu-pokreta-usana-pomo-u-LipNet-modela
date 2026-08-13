# Vizuelno prepoznavanje govora pomoću LipNet modela

Studentski projekat iz predmeta Mašinsko učenje 2. Cilj praktičnog dela je pokretanje postojeće LipNet implementacije, analiza video pipeline-a, manji eksperimenti sa rezolucijom i augmentacijama i fino podešavanje modela na srpskom delu AI-SPEAK korpusa.

Aktivni kod za Faze 0–4 sada prati VIPL-first roadmap: pinovan je tačan upstream
commit, sačuvana je licenca, GRID model/preprocessing su adaptirani minimalno,
a AI-SPEAK tok koristi VIPL face-alignment mouth frejmove, runtime discovery,
promenljivi padding i shape-audit transfera težina.

- [Novi VIPL-first roadmap](docs/analiza-i-roadmap.md)
- [Evidencija upstream odstupanja](docs/upstream-diff.md)
- [Tekst zadatka](36%20Vizuelno%20prepoznavanje%20govora%20na.txt)

## Colab notebookovi — pokretati redom

1. [`00_faza_0_upstream_restart.ipynb`](playground/00_faza_0_upstream_restart.ipynb) — pin, licenca i inventar (CPU).
2. [`01_faza_1_grid_parity.ipynb`](playground/01_faza_1_grid_parity.ipynb) — originalni VIPL i lokalni GRID inference parity (GPU).
3. [`02_faza_2_ai_speak_preprocessing.ipynb`](playground/02_faza_2_ai_speak_preprocessing.ipynb) — AI-SPEAK MP4 u VIPL mouth JPEG foldere (GPU).
4. [`03_faza_3_serbian_dataset.ipynb`](playground/03_faza_3_serbian_dataset.ipynb) — srpski parser, split, Dataset i promenljivi batch (CPU).
5. [`04_faza_4_transfer_ctc_smoke.ipynb`](playground/04_faza_4_transfer_ctc_smoke.ipynb) — srpski head, transfer audit i jedan CTC backward (GPU).

Svaki notebook je čitljiv odozgo nadole, ima parametrizovane Drive putanje i
čuva male rezultate/izveštaje na Drive. Notebookovi sa GPU radom samo pripremaju
komande; u ovom repozitorijumu nisu pokretani CPU zamenom.

## Aktivna struktura

```text
lipnet/                  # VIPL model, Dataset, demo/preprocessing i CTC pomoćni kod
scripts/prepare_ai_speak.py
data/splits.py           # eksplicitni speaker-disjoint split
playground/00_...04_...  # Colab faze
docs/upstream-diff.md
```

Prethodni `app/*`, `lipnet_faza1_colab.ipynb`, `lipnet_faza2_gpu_colab.ipynb` i
Faza 1/2 dokumenti su legacy materijal. Novi tok ih nigde ne uvozi i ne koristi
njihove manifest/ROI artefakte.

Profesor je obezbedio `processed.zip` kao ulazni korpus. Arhiva, raspakovani
`processed/` podaci, generisani mouth frejmovi i težine modela ostaju van Git-a.
Notebook Faze 2 očekuje tu postojeću arhivu na Google Drive-u; ne preuzima
privatni korpus, već ga lokalno pretvara u VIPL-kompatibilne JPEG sekvence.
