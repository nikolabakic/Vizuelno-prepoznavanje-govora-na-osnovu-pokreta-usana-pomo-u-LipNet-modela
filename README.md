# Vizuelno prepoznavanje govora pomoću LipNet modela

Studentski projekat iz predmeta Mašinsko učenje 2. Projekat reprodukuje pinovanu
VIPL LipNet implementaciju, prilagođava je srpskom delu AI-SPEAK korpusa i meri
uticaj rezolucije, blur-a i pomeranja mouth crop-a.

Aktivni tok je zasnovan na VIPL commit-u
`40209e09c49553c00c25c7d41faa3706aea3c625`. Veliki i privatni podaci ostaju
na Google Drive-u; u Git-u su samo kod, prazni reproduktivni notebookovi,
testovi i dokumentacija.

- [Roadmap i kriterijumi](docs/analiza-i-roadmap.md)
- [VIPL odstupanja](docs/upstream-diff.md)
- [Audit postojećih rezultata](docs/provera-rezultata.md)
- [Originalni tekst zadatka](36%20Vizuelno%20prepoznavanje%20govora%20na.txt)

## Status rezultata

Length-aware Faze 4–6 završene su na NVIDIA L4 nad zamrznutim
`ai_speak_lip.zip` artefaktom. Potvrđeni baseline na 540 speaker-disjoint test
uzoraka ima WER `0.4524691358`, CER `0.1823867262` i sentence exact match `0.0`.
Faza 6 je pre eksperimenata bit-po-metrici reprodukovala taj rezultat.

Aktivna verzija:

- maskira padded vreme posle svakog 3D CNN bloka;
- koristi `pack_padded_sequence` u oba BiGRU sloja;
- računa corpus-level WER/CER iz ukupnih edit grešaka i ukupnog broja
  referentnih reči/karaktera;
- koristi Drive folder `phase5_length_aware_v2` i checkpoint SHA-256
  `203c2707b5c327c8b164ab573f5550390def3aacf0ff190fc9bd760745e2f9c8`;
- u Fazi 6 ponovo računa baseline i zahteva tačno poklapanje sa Phase 5
  rezultatom pre eksperimenata.

Sanitizovani, mali JSON artefakti su u [docs/results](docs/results), a istorija
starih nevalidnih run-ova i objašnjenje ispravke ostaju u
[auditu rezultata](docs/provera-rezultata.md). Osmočasovni BlazeFace preprocessing
se ne ponavlja: Faze 3–7 samo raspakuju postojeći Drive ZIP.

## Notebookovi — pokretati redom

1. [00 — upstream restart](playground/00_faza_0_upstream_restart.ipynb): pin,
   licenca i inventar (CPU).
2. [01 — GRID parity](playground/01_faza_1_grid_parity.ipynb): originalni i
   lokalni VIPL inference na istom primeru (GPU).
3. [02 — AI-SPEAK preprocessing](playground/02_faza_2_ai_speak_preprocessing.ipynb):
   MP4 u VIPL mouth JPEG foldere, checkpoint restore i QA (GPU).
4. [03 — srpski Dataset](playground/03_faza_3_serbian_dataset.ipynb): parser,
   speaker-disjoint split, potpunost parova i promenljivi batch (CPU).
5. [04 — transfer i CTC smoke](playground/04_faza_4_transfer_ctc_smoke.ipynb):
   29-klasni head, transfer audit i backward (GPU).
6. [05 — baseline fine-tuning](playground/05_faza_5_baseline_finetuning.ipynb):
   length-aware trening, resume, izbor prema validation WER-u i jedna test
   evaluacija (GPU).
7. [06 — robustnost ulaza](playground/06_faza_6_robustness_experiments.ipynb):
   ponovna baseline provera i eksperimenti rezolucije, blur-a i crop pomeranja
   nad istim checkpoint-om/test splitom (GPU).
8. [07 — CTC decoder](playground/07_faza_7_decoder_search.ipynb): greedy naspram
   prefix beam search-a bez LM-a i sa train-only karakternim 5-gram LM-om;
   validation izbor, jedna test evaluacija, bootstrap i analiza pozicija (GPU
   samo za jednokratni cache logit-a).

Notebookovi su generisani iz [scripts/build_phase_notebooks.py](scripts/build_phase_notebooks.py).
Ne menjati njihov JSON ručno; menjati generator i ponovo ga pokrenuti.

## Google Drive ulazi i izlazi

Podrazumevane putanje su:

```text
/content/drive/MyDrive/processed.zip
/content/drive/MyDrive/LipNet/ai_speak_lip.zip
/content/drive/MyDrive/LipNet/phase2_chunks_blazeface/
/content/drive/MyDrive/LipNet/phase5_length_aware_v2/
```

Ako je `processed.zip` na drugom mestu, promeniti samo `ZIP_ON_DRIVE` ili
`SOURCE_ARCHIVE` ćeliju. Faza 2 kopira arhivu u `/content` pre obrade; ne
obrađuje hiljade fajlova direktno sa montiranog Drive-a. Završno arhiviranje je
blokirano dok korisnik eksplicitno ne postavi `MANUAL_QA_PASSED = True` posle
pregleda QA slike.

## Lokalna provera

Potrebni su Python 3.12 i [uv](https://docs.astral.sh/uv/):

```powershell
uv sync --frozen --all-groups
uv run python scripts/build_phase_notebooks.py
uv run pytest -q
```

GPU, AI-SPEAK podaci i Google Drive nisu potrebni za 33 lokalna testa. Testovi
pokrivaju parser/vokabular, split, promenljivi batch, strict transfer,
checkpoint restore, corpus metrike, checkpoint round-trip i invariancu kraćeg
klipa kada je sam ili u padded batch-u, kao i egzaktno CTC prefix beam
dekodiranje, 5-gram LM, paired bootstrap i analizu grešaka po pozicijama.

## Struktura

```text
lipnet/                         VIPL model, Dataset, preprocessing i trening
scripts/prepare_ai_speak.py     MP4 -> mouth JPEG + log/checkpoint/QA
scripts/build_phase_notebooks.py
data/splits.py                  verzionisani speaker-disjoint split
playground/00_...07_...         reproduktivni Colab notebookovi
docs/results/                    sanitizovani auditi i potvrđene metrike
docs/                           roadmap, upstream evidencija i rezultat audit
tests/                          CPU testovi
```

Ne slati u Git `processed.zip`, raspakovane snimke, mouth frejmove,
checkpoint-e, QA slike sa identitetom učesnika niti Drive rezultate. Pravila za
te fajlove su u `.gitignore`.
