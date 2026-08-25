# Projektne skripte

Folder `scripts` sadrži izvršne alate koji povezuju preprocessing i
reproduktivne notebookove sa modulima iz paketa `lipnet`.

## Sadržaj

| Fajl | Namena |
|---|---|
| [`prepare_ai_speak.py`](prepare_ai_speak.py) | pretvara AI-SPEAK MP4 snimke u VIPL-kompatibilne foldere mouth frejmova |
| [`build_phase_notebooks.py`](build_phase_notebooks.py) | generiše svih osam Colab notebookova u folderu `playground` |
| [`__init__.py`](__init__.py) | omogućava pokretanje skripti kao Python modula |

## Preprocessing

`prepare_ai_speak.py` pronalazi video/alignment parove, poziva detekciju i
poravnanje lica iz `lipnet.demo`, upisuje numerisane JPEG frejmove i pravi
checkpoint i QA evidenciju. Namenjen je Colab GPU okruženju.

```powershell
uv run python -m scripts.prepare_ai_speak --help
```

Finalno arhiviranje preprocessing rezultata zahteva eksplicitno potvrđen ručni
QA, kako bi se neispravni regioni usana uočili pre treninga.

## Generisanje Colab toka

```powershell
uv run python scripts/build_phase_notebooks.py
```

Generator je jedini izvor ćelija notebookova. Posle njegove izmene potrebno je
pokrenuti generator i lokalne testove:

```powershell
uv run python scripts/build_phase_notebooks.py
uv run pytest -q
```

Pregled namene svakog notebooka nalazi se u
[`playground/README.md`](../playground/README.md).
