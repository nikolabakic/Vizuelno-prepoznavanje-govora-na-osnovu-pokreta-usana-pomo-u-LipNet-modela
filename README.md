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

## Važan status rezultata

Sačuvani izlazi starih notebookova pokazali su da su preprocessing, Dataset i
transfer checkpoint-a bili pokrenuti, ali završne Phase 5 metrike nisu bezbedne
za citiranje. Stari trening je obrađivao padded vremenske korake kroz
bidirekcioni GRU, notebook nije bio izvršen redom, a CER je poređen kroz dve
različite agregacije.

Ispravljena verzija:

- maskira padded vreme posle svakog 3D CNN bloka;
- koristi `pack_padded_sequence` u oba BiGRU sloja;
- računa corpus-level WER/CER iz ukupnih edit grešaka i ukupnog broja
  referentnih reči/karaktera;
- koristi novi Drive folder `phase5_length_aware_v2`, pa ne nastavlja stare,
  nekompatibilne checkpoint-e;
- u Fazi 6 ponovo računa baseline i zahteva tačno poklapanje sa Phase 5
  rezultatom pre eksperimenata.

Zato notebookove 04–06 treba ponovo pokrenuti na GPU-u. Detalji i stari
observirani brojevi nalaze se u [auditu rezultata](docs/provera-rezultata.md).

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

GPU, AI-SPEAK podaci i Google Drive nisu potrebni za 18 lokalnih testova. Testovi
pokrivaju parser/vokabular, split, promenljivi batch, strict transfer,
checkpoint restore, corpus metrike, checkpoint round-trip i invariancu kraćeg
klipa kada je sam ili u padded batch-u.

## Struktura

```text
lipnet/                         VIPL model, Dataset, preprocessing i trening
scripts/prepare_ai_speak.py     MP4 -> mouth JPEG + log/checkpoint/QA
scripts/build_phase_notebooks.py
data/splits.py                  verzionisani speaker-disjoint split
playground/00_...06_...         reproduktivni Colab notebookovi bez stale output-a
docs/                           roadmap, upstream evidencija i rezultat audit
tests/                          CPU testovi
```

Ne slati u Git `processed.zip`, raspakovane snimke, mouth frejmove,
checkpoint-e, QA slike sa identitetom učesnika niti Drive rezultate. Pravila za
te fajlove su u `.gitignore`.
